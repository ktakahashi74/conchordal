# 第十四版: 通常offline初回Field出生の取得結果

対象: [初回登録](body-fitness-offline-birth-registration-20260927.md)と[占有条件を明示したv2登録](body-fitness-offline-birth-v2-registration-20260927.md)。`body_fitness_birth_offline` を既定falseの独立flagとして通常 `conchordal-render` へ配線し、出生機会だけ同期の代表身体評価を行った。実時間instrumentでは起動を拒否した。

## 初回の失敗を保持

初回シーンはseed 7、Sine 440 Hz、Harmonic子380–520 Hz、`consonance(...).peak()` で取得した。RhaiのConsonance範囲指定はspacingの既定値が **1.0 ERB**。候補44個の身体fitness level最大はbin 288（440 Hz、level 0.814005）だったが、そのbinは環境Voiceの基音440 Hzに占有される。1.0 ERB以上離れた非占有候補はbin 311（519.4866 Hz）だけで、実際の選択はbin 311、最終jitter周波数519.3915 Hz。テスト側が既存の非占有優先規則を省いた「全候補最大」を期待して失敗した。これはproduction選択の不整合ではなく、登録したテスト判定の誤り。初回取得の生記録は `target/body-fitness-recovery-build-20260927/runtime-birth-evidence/normal-72-1790487502174761377/`、ログは `target/body-fitness-recovery-normal-first.log`。この取得をv2成功へ転用していない。

## v2の固定条件と結果

v2は同じseed、環境、子Body、時刻、範囲、Peakで、spacingだけ明示的に0.0 ERBへ変えた。OFF/ON各二回のWAV、spawn、population、出生診断は同mode内で一致した。ONで出生診断1件、OFFで0件。ONの候補44個は全員非占有で、身体fitness level最大のbin 288を選択。中心440.0 Hzからlog jitterして実出生440.78018 Hz。実spawn eventのVoice ID 2・population ID 2・member index 0・世代0・周波数は診断および実Voiceと一致。最終Recipe identityを実Voiceから再構築して照合した。

環境の支持終端と決定時刻はどちらもsample 33280。これはframe 64を分析した結果の右端 `(64+1)*512` と、次のframe 65の出生判断開始時刻に対応する。時刻はシナリオの `wait(0.6826667)` を無理にframe 64へ丸めて付けた値ではない。支持ageは0で、登録上限4800 samples以内。現Scenarioは解析パラメータ変更と受理audio hop欠落を許さないため、reportのepoch 0は固定した単一解析構成の識別子である。再構成を跨ぐ一般的なepoch追跡の証拠ではない。

bin 288の身体levelは0.814005、旧point levelは0.880161。44候補の身体levelとpoint levelの最大絶対差は0.255523で、登録した `1e-6` を超えた。独立な全候補配列の最大計算と実選択は一致した。ただし旧point levelでも最大はbin 288。OFF/ONの子周波数は同じ440.78018 Hzで、四回のWAV SHA-256も `156e99c829fe8267cdee48598294534412f09394c1a8ec30fcae8da4749a78f9` と同じ。したがって、このシーンは身体評価値の消費と通常経路の出生を示すが、身体評価による出生位置または音声の差は示さない。

v2生記録は `target/body-fitness-recovery-build-20260927/runtime-birth-evidence/normal-537-1790487742118109317/`。ログは `target/body-fitness-recovery-normal-v2-first.log`。最終integration取得は `cargo test --test body_fitness_birth -- --nocapture` で3/3通過、記録は `target/body-fitness-recovery-focused-final.log`。Gap/Uniformは身体候補表を作らず、旧配置を使ったskip理由を各1件報告した。OFF/ONの同seed出生HzとWAVがそれぞれ一致。flagのheadless instrument拒否、metabolism/action/observationとの同時ON拒否も通過した。最初のgate試験では、metabolism検証が同時flag拒否より先に走る順序不備で失敗し、順序修正後のfocused再取得で1/1通過した。最初の失敗記録と最終ログ `target/body-fitness-recovery-gates-final.log` を保持した。

独立数値oracleは候補中心1点と実jitter後の1点について、別のToneを72hop再生し、Analysisの主観強度を得て、各binの `q*du` からscore・level・Consonance/Dissonance/Edge massを直接積分した。productionの身体評価値と一致した。全候補の独立F2再計算や長期出生過程の検証ではない。`cargo test --lib offline_birth -- --nocapture` はoracleと重複population・解析更新拒否の2/2通過、`cargo test --lib production_birth_evaluator_rejects_missing_stale_and_wrong_space -- --nocapture` は支持欠落・古い支持・空間不一致の1/1通過。ログは `target/body-fitness-recovery-lib-offline-birth-final.log` と `target/body-fitness-recovery-lib-rejection-final.log`。既存F4f focused 11/11も `target/body-fitness-recovery-f4f-focused.log` に保持した。

同期計算費用、非同期実時間出生、複数初回子、親付きrespawn、動的解析再構成、長期生態は今回の取得範囲外。全suiteとtimingは統括担当の取得を待つ。
