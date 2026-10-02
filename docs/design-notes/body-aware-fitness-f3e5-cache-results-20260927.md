# F3e5: exact 候補密度の有界再利用 取得結果

日付: 2026-09-27。状態: 隔離 worktree の `cfg(test)` 専用結果。取得前登録は `body-aware-fitness-f3e5-cache-registration-20260927.md`（SHA-256 `c529724d9552070f41e6f9a03c88ed87676a450cb5b022bccac012d3919406c2`）。source は `src/runtime/body_fitness_f3e_delivery_tests.rs`（SHA-256 `b198881654985f041d31adc2b82d6c2477475d894a0e767dd456b0f1be089c37`）、release test binary は SHA-256 `f53ffa0600c61099dff2f29c0c3e65d2b71fbd87ce47fb69822740e334cae319`。親が他のcargo/renderを停止した専有枠で、1 source と4 sourceを各一回だけ取得した。再測定による選別はない。

| 条件 | 1 source | 4 source（source順） |
| --- | ---: | ---: |
| 処理 hop／予定時間 | 938／10.005334秒 | 938／10.005334秒 |
| 実身体表消費 | 196 | 194, 193, 195, 194 |
| Ready機会 | 196 | 194, 193, 195, 194 |
| 延期 gate | 742 | 744, 745, 743, 744 |
| q cache hit | 25,284 | 25,026, 24,898, 25,156, 25,027 |
| q cache miss＝実q再計算 | 720 | 714, 710, 716, 713 |
| eviction | 464 | 458, 454, 460, 457 |
| 最大保持 entry／計上byte | 256／727,040 | 各256／727,040 |
| 窓内完了 | 197 | 194, 194, 195, 194 |
| 窓外完了 | 0 | 1, 0, 1, 1 |
| 測定終了後join回収 | 1 | 各1 |
| deadline miss | 0 | 0 |
| F3b受信／失効 | 928／0 | 937／0 |
| 最大同時保持 | 1 | 4 |
| 最大start jitter | 2.143 ms | 0.072 ms |
| test exit | 0 | 0 |

拒否理由は両条件の全sourceで空。実score利用数は1 sourceで28,318、4 sourceで28,028／27,880／28,177／28,025。初回q要求は候補132件を実計算し、所要時間は1 sourceで1.608秒、4 sourceで各1.660–1.684秒。以後の要求は共通grid候補を再利用し、新しいRNG候補だけを計算した。1 sourceの2回目以降のq所要時間中央値は37.528 ms、4 sourceは各38.111–38.383 ms。全要求の所要時間分布は生ログの `completion_durations_sec` に保存した。測定窓内の最終完了と実gate消費は異なるため、1 sourceの窓内完了197件も実消費は196回。窓外完了・最終join回収も実消費には数えない。

取得前の成功条件（各sourceの実消費がphase4の5回を超える、deadline miss 0、F3b失効0、保持上限遵守）はすべて満たした。固定440 Hz・固定Sine身体・固定Recipe・`landscape_weight=0`・`move_cost_coeff=10` に限定した結果で、延期gateはなお742–745回ある。1 sourceの最大start jitterは2.143 msだが、定義済みのhop処理deadline missは0。応答頻度は約19.3–19.6回／秒で、phase4の約0.5回／秒から改善したが、全938 gateで判断できたわけではない。

機能検査では、cache有無の密度全bin・score・実Voice判断／RNGのbit一致、通常RNG更新時の共通候補129 hitと新候補3 miss、基準周波数のみ変化した場合のhitとbit一致、recipe／body generation／source generation／birth／epoch／観測hop数／空間／front-end変更での失効を確認した。件数256のLRU touchと1 MiB以内のbyte evictionも作動した。`cargo test` は全通過し、最後の参照2箇所のClippy修正後にfocused 4件を再確認した。最終sourceの統合全suiteは親の統合worktree側で1,258成功／0失敗／45 ignore（12:21:53）と確認された。標準 `cargo clippy -- -D warnings` とfmtは通過。追加で走らせたall-targets Clippyは修正前に19件（既存17件、今回source由来2件）の警告で不通過。今回の2件は修正済みで、all-targets Clippyの再実行はしていない。

生ログと同一shellで保存した終了値:

| 条件 | raw log SHA-256 | status SHA-256 |
| --- | --- | --- |
| `targeted-logs/f3e5-cache-live-1source.log`／`.status` | `69f63e535d823487a70b579a56ed40974d345aa8af611192593db413a72e74f8` | `34ba71a94374d95233f4cdcc6fe26a5c6153c2494980d03a1d9c4350e28a2de7` |
| `targeted-logs/f3e5-cache-live-4source.log`／`.status` | `37a24626047e65d5b9e63324c6ffed3bc33cbfbca19ac3807dab33c462715623` | `97371e2623774092e4ed0c342749753b7d7d16d6c6fe5c1d972e60705ae456da` |

この測定は `cfg(test)` 専用の配送政策であり、通常runtimeへのcache配線や音楽的妥当性を示さない。実PCMの振幅は0.06、代表密度Toneは振幅1.0で、modal seedも実Toneのtone idと異なる。固定Sineではmodal seed差は作用しないが、Modalへは一般化できない。変化する身体・任意のglide・source交替中の有効率も未取得。固定current pitchに対する旧い厳格validationのまま測定した。glide時のCurrentPitchChanged扱いは別登録・別実装で検証する。phase4 capsuleは変更しない。

source・取得前登録・結果文書・raw／status・test record・release binary は `.worktrees/target-body-fitness/f3e5-cache-sealed-20260927-122403/` に封印した。`SHA256SUMS` 自体のSHA-256は `4840d436c38313083fcb1ce79522f3b17d0caced9311df650d1884729aa6ab31`、全11項目の照合は通過。この段落は封印後に元文書へ追記したため、capsuleの結果文書は追記前版を保持する。
