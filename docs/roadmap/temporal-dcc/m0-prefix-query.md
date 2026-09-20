# M0: 継続prefixの独立コピーと原音声支持の配送

2026-09-12。継続中のdescriptorから、live spanを変えずに照合用prefixをコピーする経路を実装した。
MR1配送契約の末尾欠測の扱いも修正した。いずれも`cfg(test)`内の数値経路であり、
本番query scheduler、occurrence所有、M0全体の完成ではない。
[source・検査・資源の記録](m0-prefix-query.json)を保存する。

## 独立コピー

`Span::prefix`は有界bankだけをコピーし、そのコピー上でpending blockとpending gapを閉じる。
コピーの挿入が容量を超えた場合も、live側のknot、圧縮回数、誤差、cadence、未確定block、
取得位置、最終配送時刻を変えない。元rawを再消費しない。将来のspan終端を宣言した継続状態でも、
現在までのprefixを取得できる。`finish`によるspan全体の確定は、宣言した終端への到達を要求し続ける。

コピーは元group、span始端とprefix終端、取得cut、最大元availability、最後の支持済み音声sample、
圧縮時の尺度、再構成誤差を保持する。音声支持は`Option<u64>`であり、全maskだけの入力は`None`となる。
末尾欠測でprefix終端が伸びても、音声支持をその時刻へ進めない。後の短いsource windowが、
先に使った長い音声支持を隠すこともない。尺度は圧縮時の下限処理済みの値を保存する。

live側が進んだ後、過去cutで再びprefixを要求すると拒否する。過去の照合には当時の保存コピーを使う。
コピー後の継続・圧縮やspan確定は、その保存値を変更しない。空spanにはqueryを作らない。

## 配送契約の修正

MR1の旧Ticketは`prefix終端 <= 音声支持終端 <= availability`を要求していた。
これは末尾欠測のある合法なprefixを拒否し、支持音声のないqueryも表現できなかった。
今回、prefix終端と任意の音声支持終端を、それぞれ独立してavailability以下と検査する形へ改めた。
epoch、model版、query世代、support ID、発行cut、期限、packetの完全なTicket一致は維持する。
未取得の音声終端を与えても、query終端をavailabilityより後へ進めても拒否する。

接続検査では、音声支持7,168 sample、末尾欠測込みquery終端8,192 sampleのprefixを保存した。
元spanをさらに進めた後に保存コピーを順序照合し、9,216で発行、9,728で完了、10,240で受信した。
四つのsupported候補をMR1 consumerへ渡し、元音声支持と実受信時刻を分けて確認した。
これは供給済みrawから既存照合・契約consumerへの接続であり、本番workerへの配送ではない。

## 参照照合と反例

登録Python matcherのsource hashは`cd2c2a162e542b310f33d561ec776577d3b33dbba65b103c54554ec4ad73037e`で、
現在のsourceと一致した。容量4・64・128とcadence一・二・四hopの九条件、1,408操作を使い、
54 queryを比較した。圧縮後・pending・部分取得・末尾gapを含むknot内の比較対象f64値はbit一致した。
各queryの前後でlive bankと未確定状態が変わらない条件も検査した。fixture再生成もbyte一致した。

最初の接続検査は、一knotの記憶に同じ一knotと末尾欠測を持つcueを与え、supportedを期待して失敗した。
登録DTWは同点で対角を優先するため、この例では観測stepを挿入し、欠測stepを記憶に対応させる経路を選ぶ。
PythonとRustはともに総cost 1、観測一step、対応座標ゼロ、unknownとなった。
失敗ログと独立Python出力を残し、Rustにもこの反例を固定した。対応があるはずという期待で
unknownを成功へ変えず、配送接続の正例には二knot以上の支持を持つ別入力を使った。
この短い入力での結果は、知覚的な再認失敗や全欠測入力の一般則を意味しない。

## 資源と残件

prefix作成は、コピーbankと挿入scratch、独立したFrozen payloadを明示的に確保する。
packed exportとmatcher配列にも別の確保がある。これらをhop更新の無確保という説明に含めず、
O04のcopy・allocator費用へ計上する。Ticketの任意音声終端とFrozenのmetadata追加により、
以前の型サイズを現在のサイズとして再利用しない。

600回のwarmup後、同じ満杯bankとpending blockから6,000回コピーしたrelease単体測定は次の通りである。
copy時間はコピー側の圧縮を含む。export時間はpacked bytesとmatcher配列への変換で、照合計算を含まない。

| 容量 | copy p99 µs | copy最大 µs | export p99 µs | export最大 µs |
|---|---:|---:|---:|---:|
| 64 | 0.92 | 3.00 | 2.37 | 13.93 |
| 128 | 1.83 | 8.681 | 5.11 | 7.59 |

容量128では、copy bank payload 40,960 bytesとheader 456 bytes、Frozen payload 40,960 bytesと
header 176 bytesを使う。別途packed export 40,960 bytes、matcher rows 21,504 bytesと
local interval 2,048 bytesがある。各exportの寿命は別なので、単純合算を実測peakとはしない。
この環境のMR1型はTicket 104、Packet 3,696、Receipt 136、Controller 6,296 bytesだった。
これは同じbankを繰り返し読む単体測定で、実streamの全負荷、allocator overhead、RSSの測定ではない。

全query scheduler・cache、occurrence／contextのspan所有と確定credit、全beamとjoint inference、
全負荷O04、実波形の回収・fit・校正、人の素材・分割・機関手続き・pilot・検出力は残る。
仕様本文、登録fit座標、演奏経路は変更していない。

## 再現

最終sourceの全Rust検査は894成功・0失敗・28 ignored。今回追加した通常検査は六件である。
Python matcher参照33件、releaseのprefix関連八件と明示実行した費用probeも成功した。
三つの数値集計はdebugとreleaseで一致し、短い末尾欠測の反例は独立Python出力とも一致した。
通常clippyは`-D warnings`、全targetでは既存の`unnecessary_cast`／`manual_is_multiple_of`のみを除外して通過した。
formatとdiff検査も通過した。

```bash
cargo test --lib prefix -- --nocapture
python3 -m unittest discover -s tests -p test_evaluate_temporal_matcher_reference.py
python3 scripts/generate_temporal_prefix_fixtures.py --output /tmp/prefixes.json
cmp tests/fixtures/temporal_cognition/prefixes.json /tmp/prefixes.json
cargo test --release --lib prefix -- --nocapture
cargo test --release --lib prefix_copy_cost_probe -- --ignored --nocapture
```
