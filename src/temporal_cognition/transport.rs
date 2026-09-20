//! MR1's bounded contract consumer. No model representation crosses this boundary.

#[cfg(test)]
pub(super) const ACTIVE_LIMIT: usize = 256;
#[cfg(test)]
pub(super) const RESULT_LIMIT: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Identity {
    pub id: u64,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Relation {
    pub identity: Identity,
    pub supported: bool,
    pub ambiguous: bool,
    // Log2 frequency shift and log2 reference/cue interval ratio; never probability.
    pub transformation: [Option<f64>; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Ticket {
    pub bus: u8,
    pub epoch: u64,
    pub model_version: u64,
    pub query: Identity,
    pub support_id: u64,
    pub support_start: u64,
    pub support_end: u64,
    pub supporting_audio_end: Option<u64>,
    pub available_at: u64,
    pub issued_at: u64,
    pub deadline: u64,
}

#[derive(Clone)]
pub(super) struct Packet {
    pub ticket: Ticket,
    pub completed_at: u64,
    pub relations: Box<[Option<Relation>]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rejection {
    Invalid,
    Busy,
    Stale,
    Retired,
    Capacity,
}

pub(super) struct Receipt {
    pub ticket: Ticket,
    pub received_at: u64,
    pub supported: usize,
    pub unknown: usize,
    pub ambiguous: usize,
}

pub(super) struct Controller {
    bus: u8,
    epoch: u64,
    model_version: u64,
    pending: Option<Ticket>,
    highwater: Option<u64>,
    last_handle_id: u64,
    active: Box<[Option<Identity>]>,
}

impl Controller {
    pub fn new(bus: u8, epoch: u64, model_version: u64, capacity: usize) -> Self {
        assert!((1..=super::memory::MAX_EPISODES).contains(&capacity));
        Self {
            bus,
            epoch,
            model_version,
            pending: None,
            highwater: None,
            last_handle_id: 0,
            active: vec![None; capacity].into_boxed_slice(),
        }
    }

    pub fn register(&mut self, identity: Identity) -> Result<(), Rejection> {
        if identity.id == 0 || self.active.iter().flatten().any(|x| x.id == identity.id) {
            return Err(Rejection::Invalid);
        }
        if identity.id <= self.last_handle_id {
            return Err(Rejection::Stale);
        }
        let slot = self
            .active
            .iter_mut()
            .find(|slot| slot.is_none())
            .ok_or(Rejection::Capacity)?;
        *slot = Some(identity);
        self.last_handle_id = identity.id;
        Ok(())
    }

    pub fn retire(&mut self, identity: Identity) {
        for slot in &mut self.active {
            if *slot == Some(identity) {
                *slot = None;
            }
        }
    }

    #[cfg(test)]
    pub fn restart(&mut self, epoch: u64, model_version: u64) -> Result<(), Rejection> {
        if epoch <= self.epoch {
            return Err(Rejection::Stale);
        }
        *self = Self::new(self.bus, epoch, model_version, self.active.len());
        Ok(())
    }

    pub fn dispatch(&mut self, ticket: Ticket, now: u64) -> Result<(), Rejection> {
        if ticket.bus != self.bus
            || ticket.epoch != self.epoch
            || ticket.model_version != self.model_version
            || self.highwater.is_some_and(|old| ticket.query.id <= old)
        {
            return Err(Rejection::Stale);
        }
        if !(ticket.support_start < ticket.support_end
            && ticket.support_end <= ticket.available_at
            && ticket
                .supporting_audio_end
                .is_none_or(|end| end <= ticket.available_at)
            && ticket.available_at <= ticket.issued_at
            && ticket.issued_at == now
            && now <= ticket.deadline)
        {
            return Err(Rejection::Invalid);
        }
        if self.pending.is_some_and(|pending| now <= pending.deadline) {
            return Err(Rejection::Busy);
        }
        self.pending = Some(ticket);
        self.highwater = Some(ticket.query.id);
        Ok(())
    }

    pub fn cancel(&mut self, ticket: Ticket) {
        if self.pending == Some(ticket) {
            self.pending = None;
        }
    }

    pub fn receive(&mut self, packet: &Packet, now: u64) -> Result<Receipt, Rejection> {
        if packet.relations.len() > 4 * super::memory::MAX_CANDIDATES {
            return Err(Rejection::Capacity);
        }
        if self.pending != Some(packet.ticket) || now > packet.ticket.deadline {
            return Err(Rejection::Stale);
        }
        if packet.completed_at < packet.ticket.issued_at || packet.completed_at > now {
            return Err(Rejection::Invalid);
        }
        let mut receipt = Receipt {
            ticket: packet.ticket,
            received_at: now,
            supported: 0,
            unknown: 0,
            ambiguous: 0,
        };
        for relation in packet.relations.iter().flatten() {
            if !self.active.contains(&Some(relation.identity)) {
                return Err(Rejection::Retired);
            }
            if relation
                .transformation
                .iter()
                .flatten()
                .any(|x| !x.is_finite())
            {
                return Err(Rejection::Invalid);
            }
            receipt.supported += usize::from(relation.supported);
            receipt.unknown += usize::from(!relation.supported);
            receipt.ambiguous += usize::from(relation.ambiguous);
        }
        if packet.relations.iter().all(Option::is_none) {
            receipt.unknown = 1;
        }
        // Validate the entire packet before consuming the ticket or exposing effects.
        self.pending = None;
        Ok(receipt)
    }
}
