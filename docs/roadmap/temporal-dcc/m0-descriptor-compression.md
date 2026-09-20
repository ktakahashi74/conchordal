# M0: 原支持を保つdescriptor圧縮と順序照合への接続

2026-09-12。登録済みのmoment圧縮をRustへ移し、音響Frontendのraw descriptorから
既存の順序付き照合へ接続した。実装は`cfg(test)`の数値経路に限る。
[source・検査・費用の記録](m0-descriptor-compression.json)を保存する。M0全体は未完了である。

## 圧縮と取得支持

各knotは320 bytesで、10座標の重み・平均・中心二次momentをf64で保持する。
代表終端時刻、取得時間、span支持、元音声支持、availability、欠測時間、epochと世代も保存する。
既定の二hop cadenceはgroup世代の最初のhopに合わせる。spanの端と欠測で部分blockを閉じ、
連続欠測は一つの全mask blockへまとめる。欠測からの再開は、旧部分block・欠測・現在blockの
最大三挿入となる。

部分取得をspanへclipするときは、取得sample数だけでなく、元の取得区間の位置を使う。
同じ取得数でも位置の異なる再送を拒否する。正規化した取得位置のbitmapと元raw値のbit表現で
完全な再送を区別し、重複配送で重みを増やさない。別bus・epoch・世代や未来のavailabilityも拒否する。
span内の評価支持をclipしても、元NSGT窓とavailabilityを短縮しない。

容量128を超える挿入では、先頭と新しい末尾を保護し、旧内部pairのうち尺度付き再構成誤差の
増分が最小のpairを統合する。同値なら先の支持を選ぶ。統合は重み付き中心momentで行い、
平均の大きい入力で二つの大きな二乗値を差し引かない。尺度の下限は登録通り1e-6である。
90%未満の座標支持はmaskし、欠測を含むknotからtempo支持を作らない。

確定時は宣言したspan終端への到達と、全blockの元availabilityを覆うsample cutを検査する。
末尾より前のblockに最大availabilityがある場合も保持する。確定結果は独立したコピーであり、
元spanは以降の観測を消費しない。既存matcherへのexportは128 knotまでとし、容量256の感度対照を
無言で切り詰めずに拒否する。取得済みの隣接knot間だけにlocal intervalを作る。

## 数値検査

登録Python版と現行Python版の`DescriptorKnot`、`BoundedDescriptor`、`SpanDescriptor`は
ASTで同一だった。両sourceを保存し、既存の登録source hashは変更していない。
入力浮動小数点値はIEEE754のbit表現でfixtureへ保存する。JSON十進数の再解釈が
近接した圧縮順位を変える影響を、この実装比較へ混入させない。

容量3・8・64・128・256、cadence一・二・四hop、欠測、部分取得、span clipping、同順位、
平均1e8付近の入力など20条件・10,481操作を比較した。145 checkpointでknot、pending、
欠測、圧縮回数、優先度計算数を照合し、比較対象のknot内f64値はbit一致した。
累積再構成誤差の最大絶対差は0.0001220703125で、尺度下限を使う大きな誤差値の加算順序に由来する。
検査では128 epsilonの相対・絶対許容幅を用いる。fixtureの再生成はbyte単位で一致した。

平均1e8付近の300入力を容量8へ圧縮した独立検査では、元入力から直接求めた二乗誤差は
2.336751999918649、記録した診断値は2.3367543667303408、差は約2.37e-6だった。
累積診断値は浮動小数点で計算した再構成誤差であり、一般的な誤差上界ではない。
非有限のmomentや誤差は拒否する。内部の数値エラーは部分flush後にも起こり得るので、
そのspanを捨てて扱い、同じspanへの再試行で回復できるとはしない。

別々に初期化した二つのFrontendへ、同じpower scan・energy系列を異なるsample時刻で入力した。
各317 raw hopを128 knotへ圧縮し、先のtraceをepisode、後のtraceをqueryとして既存照合へ渡した。
最良costは約2.68e-32、適用変換はpitch・tempoとも0で、7,875 DP cellを評価した。
これは特徴抽出から照合までの契約検査であり、実波形の回収や知覚的再認の検査ではない。

## 資源と未完了範囲

bank本体は容量×320 bytesを構築時に確保する。pending二つと挿入scratch一つ、
struct metadata、取得位置bitmapは別に必要となる。hop 512では二つのbitmapのpayloadは
合計128 bytes／spanである。通常更新ではこれらを再利用する。
確定コピー、packed bytes、matcher用の配列exportは明示的な別確保であり、全負荷のcopy費用から除外しない。

releaseで600 hopのwarmup後、6,000更新を測定した。二hop cadenceなので3,000回はblockを閉じる更新である。

| 容量 | bank payload bytes | 全更新p99 µs | blockを閉じる更新p99 µs | 確定コピー一回 µs |
|---|---:|---:|---:|---:|
| 64 | 20,480 | 1.41 | 1.42 | 2.15 |
| 128 | 40,960 | 1.49 | 1.53 | 4.38 |
| 256 | 81,920 | 2.88 | 2.90 | 7.86 |

この環境のSpan structは1,616 bytesで、pending・scratch・metadataを含む。
bank payloadと128-byte bitmapはこのstructの外に確保する。確定コピーは一回の測定でありp99ではない。
容量64の最大更新は20.30 µs、128は3.15 µs、256は5.28 µsだった。allocator overheadやRSSはこの表に含まない。

今回の単体費用は一つのspanに供給済みraw descriptorを与えた測定である。
全1,024 span、raw抽出、beam、照合、二worker、64 Voice、実出力機器を含むO04を代用しない。
実際のoccurrence／contextによるspan所有とcredit、全beam・joint inference、本番接続、
実音尺度のfit・校正・回収、人の素材・分割・機関手続き・pilot・検出力の条件は残る。

## 検証と再現

全Rust検査888成功・0失敗・27 ignored。今回の通常検査は七件、単体費用probeはignoredである。
releaseのdescriptor関連八検査と明示実行した費用probeも成功し、三つの数値集計はdebugとreleaseで一致した。
既存Python descriptor検査30件も成功した。通常clippyは`-D warnings`で成功し、全targetでは
従来からの`unnecessary_cast`／`manual_is_multiple_of`のみを除外して成功した。
仕様本文、fit座標、既存の演奏経路は変更していない。

```bash
cargo test --lib descriptor -- --nocapture
python3 -m unittest discover -s tests -p test_evaluate_temporal_descriptor_reference.py
python3 scripts/generate_temporal_descriptor_fixtures.py --output /tmp/descriptors.json
cmp tests/fixtures/temporal_cognition/descriptors.json /tmp/descriptors.json
cargo test --release --lib descriptor -- --nocapture
cargo test --release --lib descriptor_compression_cost_probe -- --ignored --nocapture
```
