# 実音観測: frame内のピークと保存的エネルギー配分

2026-09-12。状態: M1の観測経路に最初のtrajectory数値処理を追加。
O03／O04／O09の一部であり、M0の完了ではない。
[記録](m1-trajectory-input.json)にsource、入力、結果、検証出力を保存する。
前段の[観測実装](m1-observation.md)と[評定fit準備](m0-ordinal-fit-preparation.md)の
archiveは変更せず、各時点の証拠として保持する。

## 入力と配分

`temporal_mode("observe")`で両busの解析workerがcanonical hopのmono平均二乗
`sum(y[n]^2)/N`をf64で計算し、NSGTのLog2 power scanと同じframe packetへ渡す。
この値の支持区間は`[hop_start_sample, available_sample)`である。
スペクトルの支持区間`[support_start_sample, support_end_sample)`とは長さが違う。
device出力のチャンネル数を分母へ使わない。

専用observerは原仕様§9.2のframe内ピークを最大7件抽出する。局所最大は左より厳密に大きく、
右以上とする。frame最大の1%以上、両側2 bin先の大きい値に対するprominenceが自身の10%以上
という条件で、plateauの先頭だけを候補とする。端では存在する側を使い、両側とも2 bin先が
存在しない小さいgridではprominenceを観測できないので、その候補は採用しない。
エネルギー降順・bin ID昇順で7件を保持する。

各binは2 bin以内の最近ピークへ一度だけ割り当てる。距離同点は小さいピークbinを選び、
強度順位の先頭へ寄せない。該当ピークがなければ残差へ割り当てる。
8枠の配分エネルギーは`E_bus * assigned_spectral_mass / total_spectral_mass`。
log-envelopeは`0.5*log2(max(E_i,1e-12))`とする。

スペクトル質量ゼロでsample energyが正なら、その全量を残差へ保持し、スペクトル形状を欠測にする。
既知の無音は全枠の既知ゼロ、log-envelopeはfloor値になる。source gap後のNSGT warmup中は
ピーク・配分全体がunknownであり、新しいhopのmono energyが観測済みでも代入しない。
非有限値・負エネルギーはpacket拒否として数え、既知ゼロへ置換しない。

## 出力と境界

workerは固定長`TrajectoryFrame`を共通のUI/report snapshotへ渡す。
`peak_bins`の7枠は強度順位、`energy`と`log_envelope`の末尾は残差である。
空いたピーク枠はエネルギーゼロで、実在するtrajectoryのIDを持たない。
GUIは候補数と残差エネルギーを同じsnapshotから表示する。

これらはframe内の候補である。ridge連続性、世代handle、group assignment、分裂・結合、
accent・関係推論の実装ではない。配送欠落をまたいだ連続性も主張しない。
`relations_implemented=false`、`action_enabled=false`を維持する。
内部packet版を2に進めたが、fit用の既存raw featureとordered layoutは変更していない。
そのためfeature登録版30を維持し、今回の実装契約を別項目`trajectory_frontend`に記録する。

ピーク順位の更新は固定7枠、割当は各binに最大7比較。計算はLog2 bin数に線形で、
kernel自身はheap allocationを追加しない。返却値は実測248 bytesであり、共有snapshotの
固定サイズは増える。mono energy計算は解析threadに1回のhop走査を追加する。
queue、既存のscanコピー、snapshot lock、report serializationの費用は単体kernel測定に含めない。

## 検証と残る工程

- 独立したPythonの有理数参照106入力をRustと照合した。ゼロ、閾値、plateau、端、同点、
  7候補上限、複数gridと全量保存を含む。再生成したfixtureはbyte単位で一致した。
- source gap後のwarmup、無音、非有限値・負値のpacket拒否を既存observer検査へ追加した。
  analysis backlogの検査では実際に渡したmono sample列から得た値と最終snapshotを照合した。
- off／observeとreport有無の4条件でWAVはbyte一致し、前段に保存したWAVとも同一だった。
  両busとも58 frameを受領し、最終reportでもエネルギー保存とhabitatの既知ゼロを確認した。
- Rust全体は777件通過・14件ignore・失敗0。通常のClippyとall-target checkが通過した。

release単体kernelでは、129／513／2049 binのflat／peakedという6条件を各6,000回測定した。
最大の条件別p99は2049-bin peakedの10.27 µs、全条件の最大単発値は26.68 µsだった。
これは短い単体費用の測定であり、O04の60秒warmup・600秒全worker同時負荷、2 bus／64 Voice、
実出力機器・underrun・RSSの合格ではない。

次に必要なのは連続性と欠測を伴うridge更新、groupingとlifecycle、実際のaccent・beam入力の生成。
1／4-bin半径、半分／倍の閾値によるdevelopment比較も残る。これらを含む全workerの非ゼロ処理を
測定するまでO04を閉じない。機関手続き、素材・分割、実音での回収・予測安定性等のM0条件も残る。
