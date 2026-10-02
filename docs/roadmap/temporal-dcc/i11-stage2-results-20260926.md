# I11 第2段・隔離実装の局所結果

日付: 2026-09-26。対象: `.worktrees/i11-stage2`、基底commit `06a4772c43d06b41b44753bb0be891f23e16b93e`。到来項の試験実装、独立数値参照、同一seedのON/OFF比較と既定OFF回帰を記録する。作者既定採用、I11-2の全要件完了、R2資源受入、I12b完了は判定しない。取得前契約は [`i11-stage2-test-registration.md`](i11-stage2-test-registration.md)。

## 保存物と検査

取得先は `target/i11-stage2-comparison-20260926-204923/`。実行binary `render` のSHA-256は `a0404e301564324351fa80e892ca9a348782fcadac798ed9e443b6412d281499`。`source-manifest.json` は基底commit、Cargo・全Rust sourceと入力の個別hashを持つ。追跡済みsource差分 `source.patch` のSHA-256は `ffc449951a7920473509cc0641521128b8a92885e0aedf15ab833fac3dafd003`、追加source一式 `source-extra.tar.gz` は `3ba889e106bda02b61efbe8a7b7a9324e751b88d81fe9318c36268f57827a0ae`。`runs.json` は3素材×ON/OFF/Noneの9本すべてexit 0。取得前に設定・scriptのhashと基準binaryを照合した。取得条件は `arrival_weight=4.0` の試験用上限で、通常既定は `arrival=false` のまま。

Python独立参照5 test、比較判定器の小例3 testは合格。RustのCDFは固定中点則で別計算した201格子点の全値を絶対差 `1e-5`、窓差分を `2e-5` で照合し、経過時間不確実・reset・期限外・48 kHz以外・Periodic階段を検査した。共通群集合、自己群のsource／身体世代／モデル版／鮮度、既知0と不明、到来による選択差、skipの基底費用反例も単体testで確認した。隔離checkoutの最終全 `cargo test` は `test_status.txt` にexit 0、通常 `cargo clippy -- -D warnings` もexit 0。`--all-targets` lintは既存テストの警告が残り不合格であり、今回追加コードの警告は修正した。

## 同一seed・3素材の作用

判定は `arrival-effect.json`。時刻・Voice IDと23候補の構造、代表身体入力、予測窓、候補の非到来項を初回選択分岐まで比較した。候補ごとの `ON.cost−ON.arrival_cost` とOFF費用は登録許容内。分岐後の履歴差は到来項だけの効果として数えない。

| 素材 | ON/OFF全決定 | 初回分岐までの共通接頭列／到来既知 | 初回選択差 | ON/OFFのWAV |
|---|---:|---:|---|---|
| sine-flow-4 | 60／60 | 60／0 | なし。群の主な除外理由は `nonperiodic` 58件 | 同一 |
| harmonic-flow-4 | 60／60 | 24／1 | `now=97792`、ON `selected_at=102136.10155710106`、OFF `100936.10155710106` | SHA不一致 |
| modal-flow-4 | 61／61 | 10／1 | `now=41984`、ON `selected_at=47631.31811746483`、OFF `45231.31811746483` | SHA不一致 |

初回分岐の2素材はskip差なし、共通入力差なし。登録した局所作用条件「3素材のうち1素材以上」を満たす。sineのCDF付きperiod群は報告上にもあるが、このrunではperiodic条件が成立せず費用に入っていない。群0件・CDF欠測は既知0に置換していない。自己群対応はこの3素材では主に `binding_absent` であり、実走による自己群除外の確認にはならない。単体testの照合成立とは別に扱う。

`binding_absent` は、現Voiceのid・generation・bus0に対応する有効なbody bindingが見つからない段階を表し、
共有音群やCDFとの照合より前に返る。静的確認ではbody captureとprototype modelは有効である。
Snapshotの配列はserializeされないが、reportには別の `body_descriptor` recordがあり、割当まで保存されていた。
取得済みの3素材ON reportを再調査し、各素材のhabitat側236件、計708件について、保存された標準化値と
prototype割当を同じ固定configから再計算してすべて照合できた。

各素材とも236件中228件が距離上限0.25超過、8件が割当ありだった。共通座標なしによる欠測はなかった。
全素材で最終captureは611 frame処理、invalid hop・capture drop・voice上限超過は0。
record層では「観測はできているが固定prototypeへ割り当たらない」ことを確認した。
初期のrecord未発行もあるため、全 `binding_absent` をこの一因へ一括帰属しない。
各decisionが消費したsnapshot版の完全な再演も、この事後調査では行っていない。

固定尺度の第6座標はアクセント頻度の `log1p` で、登録configの平均・偏差と全medoidの値が0である。
実装は偏差を `max(deviation, 1e-6)` とするため、アクセントが現れるとこの座標の距離が大きくなる。
未割当228件のうちSine185件、Harmonic188件、Modal225件では、第6座標だけの距離下限が0.25を超え、
ほかの座標にかかわらず全prototypeから拒否される。再適合した尺度・prototypeの登録が必要かを検討する材料であり、
今回の既定値や距離上限は変更していない。

この尺度と8 medoidは[I10の実音prototype登録と取得](i10-body-outcome.md#実音prototypeモデルの接続契約実装前登録)に由来する。
単音を2秒持続してreleaseする36素材・両bus・4 cutの288校正recordでは、第6座標が全件観測済みで全件0だった。
保存済み36 reportの公開`body_descriptor`計1,512件でも、この座標に正値はない。
平均・母標準偏差の0と、標準化時だけ偏差を1e-6で下限化する規則は取得前登録と一致する。
medoidのmaskは全8件で第6座標を含み、照合は共通座標が1件以上あれば距離を計算する。
したがって現flow素材の拒否は固定診断モデルの定義どおりで、計算bugとは断定できない。
校正素材とflow素材の分布差、あるいは意図された範囲外拒否の可能性を区別するには、正のaccentとゼロの双方を含む
身体・routing・bus・時刻cutを取得前固定した別の校正／保留素材、再適合版の割当率と距離分布、下流の自己群除外の
作用を比較する必要がある。現版の閾値・mask・尺度・medoidは変更しない。

再調査のsource・summary・元reportとconfigのSHAは `target/i11-stage2-binding-audit-20260926/` に保存した。

登録済みの規則では自己群対応が不明なら群を除外しない。そのため、この局所作用を「他者の到来だけを使った作用」とは呼ばない。
F3のsource別PCMによる周波数地形の自己除去は、I11の音群とsourceの対応づけとは別の機構であり、
F3の数値一致からこの自己群除外の成立を推定しない。

## 既定OFFの基準回帰

元の `config-none.toml` と同じ3つのscriptで、現隔離版のNoneを `06a4772` の `target/i11-same-record-20260926-193414/` のNone出力に比較した。`bit-identity.json` は3素材すべてWAVと学習record 10種がbit一致、`pass=true`、欠落0。`candidate-records.json` では共通keyの候補recordがsine 161、harmonic 68、modal 72件で内容差0。片側だけの候補recordは順に基準／現版が29／32、10／10、13／13件であり、配送飽和に関わる別計数として保持する。この比較は旧I11-1の登録12条件すべてを再実行した結果ではない。

## 残る境界

元の §4.5 は群ごとの「表なし」の内部原因まで求める。現snapshotでは不確実な経過時間、積分失敗、その他のCDF未生成を分けられず、報告は `cdf_unavailable` に留まる。実走で有効な自己群Bindingが得られる条件、登録比較の広い母集団、§5.1〜5.3の全独立再計算、I11-1の現版§5.7費用、R2・A1は未完。結果を係数適合や可聴受入へ拡張しない。

## 2026-09-27: 診断版の独立検算と実身体入力の追加取得

上記「CDF未生成の内部原因を分けられない」は最初の比較版の状態である。後続の診断版は原因を生成側から配送しており、main checkoutの`target/i11-stage2-diagnostic-v2-20260926/manifest.json`には9本のWAV、成功CDF値、学習record10種、共通keyの候補recordの一致が保存されている。片側だけの候補recordは別計数であり、全report一致を意味しない。

今回の[独立検算記録](../../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-independent-record-audit-20260927.md)では、保存された3素材ON/OFFの計362判断・328個のCDF表と、同じ診断binaryで新規取得したHarmonicのbody+arrival 60判断・112個のCDF表を照合した。CDFは独立した中点積分で全201点を再計算し、最大差はそれぞれ`4.1925e-8`、`2.8806e-8`。候補の変位、Hellinger項、重なり、到来費用、総費用、選択、skipの基底費用、記録内の利用可能時刻も再計算し、検算エラーは0だった。NaN、配列長不一致、空の証拠、未来入力、費用改竄などの反例を含むPython検査14件が通過した。

新規取得は事前にscene/config/binary/commandのhashを固定した。body設定は元ON設定のfootprint指定だけを変更し、係数・尺度・medoid・閾値を再適合していない。新取得の59/60判断が実身体footprintを使い、そのうち10判断で到来項も既知だった。元proxy取得との初回選択差まで11判断の非footprint入力は一致し、sample44032のVoice3でbody offset0、proxy offset1となった。ただしこの分岐時の到来群は0である。分岐後のbody＋到来既知10件は両入力の併用例であり、合同の因果効果の証拠に数えない。

通常reportのsnapshotは間引かれているため、各判断が消費した正確なsnapshotから到来確率を再計算したとは言えない。source window-start期限、通常実行で自己群を除外できる正例、全§5.1〜5.3、資源・作者受入は残る。今回の検算は`src/`を変更せず、隔離版のPython検査器と追加取得を残した。旧取得を上書きせず、I11-2全体の完了とは判定しない。

## 2026-09-27: 消費した到来入力の記録と再計算

直前の検算で残った正確なsnapshotの欠測は、後続の[別登録](../../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-consumed-arrival-registration-20260927.md)で隔離版にreport-onlyの記録を加えて検証した。[結果](../../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-consumed-arrival-results-20260927.md)では、到来費用が使う同じ参照から7群の状態・CDF全201点・採用集合・自己群照合入力・由来と期限を保存する。通常の間引き観測へ事後joinした結果ではない。既定値、モデル、係数、RNG、選択則は変更していない。

3素材proxy ONの181判断とHarmonic body ONの60判断、計241判断を正確な消費入力から独立再計算した。到来既知23判断の全529候補確率は記録CDFからの補間値との差0、CDFモデル・期限・群選択・費用・最小候補・skipを含む監査エラー0。OFFの181判断も監査エラー0で、到来provenanceと費用項はない。None3本は判断0・provenance0という別条件で照合した。旧ON/OFF/None9本と既存body ON1本の計10本すべてで、WAV hashとprovenance欄だけを除いた判断recordが一致した。

実reportに対するCDF値・handle・発行時刻・採用集合・候補確率・snapshot期限・owner・bindingの8種改竄はすべて拒否された。Pythonの反例検査8件が成功。Rust全検査は1170成功・0失敗・36 ignore、`cargo test exit=0 @ 2026-09-27T18:22:03+09:00`。fmt、標準Clippy、all-targets checkも成功した。主担当は取得plan、binary、10本のWAV/report、最終検査器のhashと368ファイルのsource capsuleを現物照合し、全ON/OFF監査とPython8件を再実行して一致を確認した。

今回の到来既知23判断はすべて採用群1つであり、複数群平均の通常正例は0。自己群Bindingが成立して除外される通常正例も0のまま。記録は消費参照との整合を検証するもので、snapshot生成元の外部認証ではない。body＋到来既知10件は併用例であり、前節の因果的限界を変更しない。自己群正例、複数群、資源・作者受入、I11-2全体は引き続き未完。

## 2026-09-27: 通常rendererの複数群取得

続く[複数群の事前登録](../../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-multigroup-registration-20260927.md)と[結果](../../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-multigroup-results-20260927.md)では、凍結binary・モデル・係数を変えず、habitatの二つの周期刺激とpresentation専用の参加者を使った。刺激は群形成を操作する明示的な研究変数であり、自律集団の自然な群形成や自己群除外の証拠にはしない。元登録のpulse-lock説明を取得後に訂正したが、scene・binary・設定・rawは変更していない。

通常取得2回とも28判断中26判断が到来既知、19判断が異なるhandleの複数群を採用した。群数0/1/2/3/4は各2/7/4/7/8判断。各回の複数群19判断・437候補の算術平均を独立再計算し、最大差`2.7756e-17`、全判断の監査エラー0だった。完全な判断配列とWAVは二回一致。主担当も入力hash、監査、判断配列・WAV一致を再実行して確認した。2音源から最大4群が生じているため、音源と群の一対一対応や到来事象の独立性は確立していない。スクリプト指定2/3 Hzに対し、観測onset間隔の中央値は約0.50149/0.37501秒であり、指定値を実測周期と同一視しない。

この取得で自己群除外は0のままだが、presentation専用Voice3にbus0のBindingが27判断で存在した。静音Recordは`active=true`でもよく、スペクトルの証拠がない部分座標でもprototypeへ対応するためである。早期のdescriptorにはmask44があるが、sample88576の判断が消費したRecordはend83968・mask60であり、両者を混同しない。この消費Recordでもスペクトルのmask bits 0/1は欠けている。`active`は記録の有効性であり音響的活動の証拠ではない。今回の対応先はCDF欠測で実際には除外されず、誤除外の実害を観測したとは扱わない。

後続の[自己群証拠の登録](../../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-self-group-evidence-registration-20260927.md)では、I11自己群除外に限り、判断時のhabitat routingと同一ownerの消費Recordにあるスペクトルの証拠・identity・時刻を必要条件へ追加する。Record.activeやprototype一般の意味は変更しない。これは実音との対応の必要条件であり、音響群の因果的な帰属の十分条件ではない。局所反例と通常回帰、独立検算の結果は次節に記録する。既存の自己群正例、資源・作者受入、I11全体の未完状態は維持する。

## 2026-09-27: 自己群除外に必要なrouteと消費Recordの検査

[隔離版の結果](../../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-self-group-evidence-results-20260927.md)として、登録した必要条件を実装した。現在のhabitat routing、同じowner/bus/body generationのRecord、Bindingとのend/available一致、既存の鮮度、スペクトルのmask bits 0/1と既存coverage条件を要求する。一般のRecord.active、prototype対応、CDF生成と係数は変更していない。[取得plan](../../../.worktrees/i11-stage2/target/i11-stage2-self-evidence-20260927/plan.json)にsource capsule、binary、入力、旧比較対象と全11コマンドを固定した。

ON5本269判断とOFF3本181判断の監査エラー0。None3本は判断0という別条件で確認した。旧版との11本すべてでWAVと、新しいprovenance・自己群理由を除く判断内容が一致した。通常3素材およびHarmonic body ONの計12判断はスペクトル証拠不足、複数群場面の28判断はhabitat未接続として自己群候補から除外された。どちらも旧版ですでに有効自己群を除外していなかったため、選択や音声の変化はない。通常の自己群除外正例は0を維持する。

全Rust検査1171成功・0失敗・36 ignore、`cargo test exit=0 @ 2026-09-27T19:06:43+09:00`。fmt、標準Clippy、all-targets check、関連Python42検査が成功した。主担当も541ファイルのsource capsuleと現物、planの15入力hash、全ON/OFF監査、11本の音声・判断比較、局所Python10検査を照合した。

この検査は消費tupleの内部整合と、自己群を除外する必要条件を検証する。maskが正しく生成されたことは既存producer契約に依存し、記録そのものの一般的な改竄検出や、音響群の因果的帰属を保証しない。route falseのままmaskだけを有効値へ改変しても自己群除外は起きず、そのような写しを必ず拒否する監査ではない。現在routeと古いRecord窓の連続性、通常自己群正例、資源・作者受入、I11全体は残る。

[正例欠測の診断](../../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-self-positive-gap-20260927.md)では、habitat接続の241判断中221判断がスペクトル証拠を持つ一方、その221判断にはBindingが一つもなかった。凍結medoidとの独立距離計算でも全件が既存閾値0.25を超える。したがって現在の欠測は、新しい自己群条件の手前にあるprototype対応で生じている。独立訓練素材とholdoutを分ける次案は未取得であり、取得済みrawへ再適合していない。
