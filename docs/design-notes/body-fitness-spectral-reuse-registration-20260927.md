# 代表身体スペクトル処理の再利用登録

日付: 2026-09-27。対象は `.worktrees/body-fitness-recovery`。v13b の非 Sine paced 取得は Harmonic の身体変更後、frame 432 までに新世代の判断を消費できず失敗した。失敗記録と判定条件は維持する。今回の変更は代表 Tone と NSGT の意味を変えず、解析用の一時バッファ確保と不要な Landscape 再構築を減らす。

`SpectralFrontEnd` の raw density、主観強度の差分、出力、peak 抽出用領域を再利用する。既存 `process_nsgt_power` の戻り値は所有値のまま保ち、代表専用経路だけ借用した一時出力を読む。peak 抽出は既存公開関数と scratch 関数で同じ数値処理を使う。`AnalysisStream` の代表専用 reset は NSGT と spectral frontend の履歴だけを消去し、`last_landscape` の固定空間を維持する。代表計算に不要な R/H と snapshot の処理は既に省略済みであり、今回の対象にしない。

検査は変更前の通常 `process_subjective_frame` と新しい代表経路を同じ PCM で比較する。Sine、Harmonic、Modal、異なる brightness、複数候補、連続72 frame と候補間 reset を含め、各 frame の主観強度 scan と loudness mass、72 frame 平均、BodyFitness を bit 一致で照合する。72 frame、候補数、FFT、係数、phase、sample の演算順序は維持する。既存 module tests と代表計算の focused tests を実行し、統合側が全 suite を検査する。

追加の独立比較では封印 v13b の `src` と Cargo、必要 fixture を `target/body-fitness-spectral-compare-20260927/v13b-isolated/` にコピーする。封印本体は変更しない。同一の ignored 比較 test を隔離旧版と v14 に置き、production runtime の既定 48 kHz、512 sample/hop、NSGT nfft 16,384、Right alignment、Log2Space 55–8,000 Hz・96 bin/octave、coherent power、tau 10 ms、loudness exponent 0.23、reference power 1e-4 を使う。各身体 Sine/Harmonic/Modal、brightness 0.30/0.85、候補 220/330 Hz の12条件で、72 frame の Tone PCM、各 frame の主観強度 scan と loudness mass、代表関数の平均 scan と in-band mass を f32 bitで保存する。PCM と全数値 raw の SHA-256および byte一致を検査し、ログを保存する。別 source を同じ Cargo targetに切替えると旧 binaryを再利用する場合があるため、旧版とv14は別targetでビルドする。

速度は取得前に主張しない。専有枠で v14 の cold job 所要時間と frame 432 回復を再取得し、v13b raw と区別して保存する。閾値緩和と v13b 記録の上書きは行わない。
