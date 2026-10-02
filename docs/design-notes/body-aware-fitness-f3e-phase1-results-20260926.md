# F3e 第一段階: 実 Voice gate の準備結果照合

日付: 2026-09-26。状態: `cfg(test)` の固定 fixture における取得結果。通常 runtime の非同期配送や公開既定への採用を示さない。

取得前登録は `body-aware-fitness-f3e-preparation-draft.md` の封印版 v2（SHA-256 `a6e282b3e7f8591129fb52905909c43854ddc887ff61b2b131fa5973d8ecf452`）。本文コピーは `target-body-fitness/f3e-phase1-registration-20260926-141642/registration.md` に保存した。先行 v1 と失敗ログも保持した。v2 は正の scene の環境を 48 hop／支持終端 24,576 sample へ、取得前に訂正した版である。

固定した seed 1・4、1・4 source で、source PCM の実レンダリング、F3b worker による source 除去環境、共有 habituation 版 48、実 Tone の 72 hop 代表身体密度、F2 採点表を用いた。4 source は Sine 440 Hz、Harmonic 440 Hz、Modal 466 Hz、Sine 660 Hz。各 Voice に自身だけを除いた受理環境を渡し、実 `Voice` → `PitchController` の gate で準備表を消費した。計 10 Voice の表利用件数は各 144–146、候補数は 132–135。判断後の target、salience、adaptation、RNG、commit 後の基音は同じ表を直接使う F3d 対照と一致した。1 source の除去環境はゼロ、4 source の各環境は他 source を含み、共有 habituation state は非ゼロだった。

準備後の source 世代・出生、epoch、body generation、body brightness、recipe hold、route、pitch control、target、current pitch、RNG、space、habituation 版、候補 key 欠落をそれぞれ失効させた。各理由で身体表利用 0 とし、同じ変更を受けた job 無し対照の target、salience、adaptation、終了 RNG と一致した。複合失効 4 件では epoch→source、body generation→target、current pitch→recipe、RNG→space の実装上の先着理由を確認した。gate 不成立時には表も RNG も消費しなかった。

`cargo test --lib f3e -- --nocapture` は 3 件通過。最終 `cargo test -- --nocapture` は lib 1121 件成功、全 suite 合計 1205 件成功・40 件 ignore・失敗 0、`test_status.txt` は `cargo test exit=0 @ 2026-09-26T23:36:46+09:00`。`cargo fmt --all --check` と標準 `cargo clippy -- -D warnings` も通過。`cargo clippy --all-targets` は成功し、lib test の既存警告 17 件のみ残る。最終ログと source は別の phase1 capsule に固定する。

この結果は固定 Seq Tone の hold／ADSR／modulator／平滑化を前提とする。実 Voice の id、世代、body snapshot、現在基音、pitch control、route は gate で再取得したが、birth sample、body generation、epoch、habituation 版は独立 fixture 台帳から供給した。任意の phonation や pitch 以外の control 更新、通常 runtime の capture 起動条件、計算中 job の配送・期限・worker 無効化は未検証。第一段階の gate は完成済みの受理 batch と表だけを扱い、`Pending`・worker 失効・`ReceivedBatch` 拒否は第二段階へ残る。環境支持期間に合わせて `AnalysisStream::process` 後の共有 C 再計算を加えた取得前失敗の記録も capsule に保持する。
