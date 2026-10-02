# 冷候補並行準備の局所検証

[第十五版の登録](body-fitness-cold-parallel-registration-20260927.md)に従い、冷cache・2〜256候補・pitch key重複なしの場合だけ、一sourceの候補計算を最大2laneに分割した。既存worker自身が一方を実行し、補助threadは最大一つ。四sourceで同時計算threadは最大8。productionは `available_parallelism >= 8` の場合だけこの経路を選ぶ。

各laneは専用のAnalysisStreamで72hopを計算する。結果は元の候補順で結合し、単一workerでcacheへ登録する。warm job、重複候補、上限超過は従来の逐次経路。並行用の重複判定集合も対象の冷jobだけに確保する。取消は候補境界で確認し、全laneの終了後に応答する。取消された冷jobからcacheへ部分結果を残さない。thread開始失敗と補助threadのpanicは明示失敗にし、Readyを作らない。診断に実際の `compute_lanes`（1／2）を追加した。

新しい局所検査4件を通過した。

- Sine/Harmonic/Modalの全候補を、同一版に保持した逐次経路と比較。pitch key、Identity、scanの全bit、cache順序・統計・確保量、現環境score、実Voice判断とRNGが一致。warm再利用は1laneで同じ結果。
- body generationとepochの変更後はcacheを失効させて2laneで再計算。旧cache hitなし、既存上限内。
- 重複候補と257候補は1laneへ戻り、重複時のhit/missとcache順序を維持。257候補でも保持量は上限内。
- 1／2／4 sourceを同時に進め、各sourceの両laneがそれぞれ実候補を一つ計算してから取り消した。全lane終了後の取消応答、新serialでの回復、hit 0／miss 1／entry 1により部分cacheの非残存を確認した。

取消gateは試験専用channelで、開始通知を最大30秒だけ待つ。低CPU数でも試験だけは2laneを強制し、productionのCPU数条件を変えない。終了用senderはworkerより先に破棄され、試験失敗でも無期限にgateを保持しない。このtimeoutや局所試験の実行時間を実時間性能の合否には使わない。

既存の準備4件とaction 5件も通過した。局所計13件の成功は、期限frame 432や最大source数の音声資源の合格を示さない。別に全suiteと固定binaryでの専有取得を行う。

新規試験の初回compileは、Cloneを持たないVoiceを複製しようとした試験コードの誤りで失敗した。独立に同じseed・仕様のVoiceを作るよう修正した。最初のfilter誤りによる0件実行も4件成功とは数えない。成功ログと失敗ログは `/home/shafi/lwrk/conchordal/target/body-fitness-cold-parallel-build-20260927/focused-parallel.log` と `focused-initial-compile-failed.log`、各 `.status` に保持する。

その後の標準Clippyでは、Barrier importを除去した際の `#[cfg(test)]` が通常経路にも必要な同期型のimportに残り、非test libのcompileに失敗した。属性一行を削除し、初回source v15と `target/cold-parallel-validation/initial-v15/` の失敗ログを保存。修正版をv15bとして固定し、通常libのClippyと全target checkを通した。全suiteと性能取得の正本は後続の統合検証に置く。
