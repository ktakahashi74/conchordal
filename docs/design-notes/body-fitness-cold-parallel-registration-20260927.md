# 第十五版: 冷候補準備の有界並行化（取得前登録）

基準は第十四版の固定408 source、manifest SHA-256 `3be6848c779cd915002af2c48e2ce87151037814f5bb3a2db57049e8eb6d48b8`。新しい隔離worktreeは `body-fitness-cold-parallel`。第十四版の取消は1 hopで応答したが、新世代135候補×72 hopの計算に1,905,687 µsを要し、期限frame 432に対して消費frame 480となった。旧結果・未達判定は保持する。

## 実装の範囲

候補集合を確定した後の代表密度計算だけを並行化する。身体、Tone、NSGT、72 hop、候補生成順、RNG、採点式は変更しない。各候補は固定phase seedと独立した解析状態で計算できる。候補scanは元のindex順に戻し、cache登録とReady形成は単一の準備workerで順番どおり行う。音声hopやcallbackに新しい待機・thread生成を入れない。

初版はbind後のcacheが空で、候補数が既存cache entry上限以内の冷jobを対象とする。重複候補など、逐次cacheのhit/eviction順を同じに保てない条件は既存逐次経路へ戻す。warm jobは既存経路を維持する。一sourceの計算laneは最大2で、既存worker自身が一方を処理し、補助laneは最大一つ。四sourceで同時の計算threadは最大8。available parallelismが8未満の実行環境では逐次経路を維持する。これは他機器に対する性能保証ではない。

補助laneは背景の準備worker内だけで生成・joinする。各laneは専用のAnalysisStreamを持ち、結果を共有可変の解析器へ書き込まない。取消は両laneで候補境界に確認し、一候補72 hopの途中では切断しない。全laneの終了を確認してから取消応答を返し、未完jobの部分結果をReadyへ使わない。冷並行jobが取り消された場合、そのjobからのcache登録は行わない。warm逐次jobで既に完了したcacheの扱いは第十四版の規則を維持する。thread開始失敗も背景workerで扱い、Voiceへ旧結果を渡さない。

既存cacheはsourceごと256 entryかつ1 MiB以下。未commit密度は既存Readyと同じ候補数上限に収め、laneごとの解析clone、PCM、scratchの追加を別に記録する。thread数・source数の有界性と、全体メモリの実測を区別する。

## 機能検証

冷jobの逐次／並行比較で、全候補のpitch key・Identity・scan、候補順、cache挿入順と統計、現環境採点と実Voice判断・RNGを照合する。Sine/Harmonic/Modalを含め、cache再利用・身体変更・epoch変更を通す。重複候補や上限超過による逐次fallbackも確認する。1／2／4 sourceの同時要求と取消を検査し、取消後の新serial回復、旧世代のcommit禁止、cacheと保持量上限を維持する。性能の合否を局所試験のwall-clock timeoutから判定しない。

第十四版と同じfull suite、format、標準Clippy、全target checkを行い、sourceとrelease実行ファイルを固定する。機能検査では数値と判断の一致を要求する。通常offlineの決定性も維持し、同flag・同seed二回の既存検査を通す。

## 専有実時間取得

第十四版と同一の取得器、seed 17、Harmonic 220 Hz／Modal 330 Hz、10秒・938 hop、frame 300 brightness変更、frame 600 control変更、OFF／ON各一回を維持する。身体回復frame 432、control回復frame 720、継続消費frame 912、全窓2声、underflow 0、hop予算超過0、support age最大4800 samples、既存cache上限を要求する。新しい出生flagと代謝flagはOFF。実音声デバイスを使わない。

候補数は候補生成規則を維持し、結果後に削減しない。観測窓や期限も緩めない。旧版との音声軌跡・候補数が非同期決定時刻により変わる場合は、固定負荷の速度比較と区別する。失敗時にはrawと判定を保持する。長期生態、実device、通常出生と代謝の結合、非同期代謝、音色遺伝、作者採用は引き続き別の未完条件。

### 四source同時準備の資源確認（取得前追加）

追加laneが最大になる四sourceの音声hopを確認するため、既存の `normal_runtime_action_paced_without_audio_device` も同じ固定binaryで一回実行する。seed 7、Sine/Drone、1／4 source、OFF／ON各10秒の計4条件。取得器と判定式は第十四版から変更せず、全窓の生存数、消費、支持age、cache上限、underflow 0、hop予算超過0を要求する。この追加は非Sine身体回復期限の代わりではなく、最大source数での資源確認である。非Sine取得と直列に行い、各取得の前後で他のcargo・test・renderがないことを確認する。
