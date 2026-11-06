//! Native transport; web execution owns nonblocking pending buffers directly.
#[cfg(feature = "native")]
pub(crate) use crossbeam_channel::*;

#[cfg(not(feature = "native"))]
mod local {
    use std::{cell::RefCell, collections::VecDeque, rc::Rc};

    /// A buffer owned by the step state. There are no paired endpoints or waits.
    pub(crate) struct Slot<T> {
        values: RefCell<VecDeque<T>>,
        capacity: usize,
    }
    pub(crate) type Sender<T> = Rc<Slot<T>>;
    pub(crate) type Receiver<T> = Rc<Slot<T>>;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum TryRecvError {
        Empty,
        Disconnected,
    }
    pub(crate) enum TrySendError<T> {
        Full(T),
        Disconnected(T),
    }
    impl<T> TrySendError<T> {
        pub(crate) fn into_inner(self) -> T {
            match self {
                Self::Full(v) | Self::Disconnected(v) => v,
            }
        }
    }
    impl<T> std::fmt::Debug for TrySendError<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(match self {
                Self::Full(_) => "pending buffer full",
                Self::Disconnected(_) => "native transport closed",
            })
        }
    }
    impl<T> std::fmt::Display for TrySendError<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            std::fmt::Debug::fmt(self, f)
        }
    }
    impl std::fmt::Display for TryRecvError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            std::fmt::Debug::fmt(self, f)
        }
    }
    impl<T> std::fmt::Debug for Slot<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("PendingBuffer")
        }
    }
    impl<T> Slot<T> {
        pub(crate) fn new(capacity: usize) -> Self {
            Self {
                values: RefCell::new(VecDeque::with_capacity(capacity)),
                capacity,
            }
        }
        pub(crate) fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
            let mut pending = self.values.borrow_mut();
            if pending.len() == self.capacity {
                return Err(TrySendError::Full(value));
            }
            pending.push_back(value);
            Ok(())
        }
        pub(crate) fn send(&self, value: T) -> Result<(), TrySendError<T>> {
            self.try_send(value)
        }
        pub(crate) fn len(&self) -> usize {
            self.values.borrow().len()
        }
        pub(crate) fn try_recv(&self) -> Result<T, TryRecvError> {
            self.values
                .borrow_mut()
                .pop_front()
                .ok_or(TryRecvError::Empty)
        }
        pub(crate) fn try_iter(&self) -> impl Iterator<Item = T> + '_ {
            std::iter::from_fn(|| self.try_recv().ok())
        }
    }
}
#[cfg(not(feature = "native"))]
pub(crate) use local::*;
