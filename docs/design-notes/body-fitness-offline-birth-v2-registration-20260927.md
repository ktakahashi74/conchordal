# 第十四版v2: 初回Field出生の選択作用（取得前登録）

[初回登録](body-fitness-offline-birth-registration-20260927.md)のseed 7、48 kHz/512、Sine 440 Hz環境、Harmonic Entrain Sustain brightness 0.7、出生時刻 `wait(0.6826667)`、380–520 Hz、Consonance Peak、出生後 `wait(0.16)`、両bus amp 0.06を維持する。変更は子の配置に `.spacing(0.0)` を明示する一点だけ。これで44候補が既存Sineの占有判定から外れ、身体fitnessによるPeak順位が実出生binを決める。旧spacing 1.0 ERBの失敗取得は別に保持し、成功例に混ぜない。

OFF/ON各二回を独立renderし、同modeのWAV、spawn、population、出生診断の決定一致を要求する。ONの出生診断は全候補binの身体fitness score/levelと身体平均mass、旧point level、選択bin、最終Hz・Recipe identity、実analysis支持sampleと決定sampleを記録する。選択binは全候補levelの最大（同値なら低bin）と独立に照合し、少なくとも一候補の身体levelとpoint levelが `1e-6` より大きく異なることを要求する。最終VoiceのHzと実spawn event、ID/member/世代、代表Recipe hashの内部照合も維持する。環境支持ageは4800 samples以内。flag OFFで出生診断は出ない。

Gap/Uniform旧経路はskip理由を診断し、Dissonance/Edge・DensityとConsonance tensionの選択数値は既存F4f内部対照を通常production evaluator経由でfocused検査する。実時間instrument拒否とmetabolism/action/observation同時ON拒否は初回の負例を維持する。v2の取得後にseed、音色、範囲、時間、閾値を変更しない。full suiteとtimingは統括担当。
