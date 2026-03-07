use crate::life::individual::ArticulationState;

pub(crate) fn step_attack_decay_envelope(
    state: &mut ArticulationState,
    env_level: &mut f32,
    attack_step: f32,
    decay_rate: f32,
    dt: f32,
) {
    let dt = dt.max(0.0);
    match state {
        ArticulationState::Attack => {
            *env_level += attack_step * dt;
            if *env_level >= 1.0 {
                *env_level = 1.0;
                *state = ArticulationState::Decay;
            }
        }
        ArticulationState::Decay => {
            *env_level *= (-decay_rate * dt).exp();
            if *env_level < 0.001 {
                *env_level = 0.0;
                *state = ArticulationState::Idle;
            }
        }
        ArticulationState::Idle => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_attack_decay_envelope_transitions_attack_to_decay() {
        let mut state = ArticulationState::Attack;
        let mut env_level = 0.0;
        step_attack_decay_envelope(&mut state, &mut env_level, 10.0, 1.0, 0.2);
        assert_eq!(state, ArticulationState::Decay);
        assert_eq!(env_level, 1.0);
    }

    #[test]
    fn step_attack_decay_envelope_transitions_decay_to_idle_at_shared_threshold() {
        let mut state = ArticulationState::Decay;
        let mut env_level = 0.0005;
        step_attack_decay_envelope(&mut state, &mut env_level, 1.0, 1.0, 0.1);
        assert_eq!(state, ArticulationState::Idle);
        assert_eq!(env_level, 0.0);
    }
}
