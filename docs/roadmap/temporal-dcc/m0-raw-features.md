# M0: group energyからraw descriptorとaccentへの接続

2026-09-12。[音響Frontend](m0-acoustic-frontend.md)に、登録済み10座標のraw descriptorと
四hop支持のaccent detectorを接続した。実装は引き続き`cfg(test)`内にある。
稼働中のobserver worker、descriptor圧縮、accent ledger、beam初期化への接続は残る。
M0全体は未完了である。検査・source・測定の記録は[m0-raw-features.json](m0-raw-features.json)に保存する。

## 数値と支持

centroid、spread、log-RMS、正のrise、正のdecline、正のspectral flux、bus energy比、
centroidから±0.5 octaveで区切る三帯域のmass比を抽出する。座標の順序と定義は既存参照を保つ。
既知ゼロenergyではlog-RMSをfloorし、既知ゼロbusのenergy比を0とする。
massがゼロなら五つのshape座標をmaskする。欠けた成分を他成分の平均で補完しない。
descriptorのfloorは既存の1e-6を保持する。accentのfloor感度比較はdescriptor座標を変更しない。

隣接差分には同じbus／epoch／group generation、association、gridの完全支持predecessorが必要である。
差分を使ったraw descriptorは、predecessorの原支持開始・終端・利用可能時刻も引き継ぐ。
10座標はgroupごとに一度だけ計算し、spanごとの再抽出はしない。

accentはriseとfluxを別々の明示的なdevelopment尺度で標準化し、正部分の平均をsalienceとする。
四つの原hopから三つのsalienceを計算し、中央値が閾値と左値を厳密に超え、右値以上の場合だけ採用する。
中央hop全体を不確実なevent区間として保持し、右側比較の終端をevent時刻へ付け替えない。
四hopの取得、association、両成分、元のsource支持とavailabilityが揃わなければ採用しない。
キャッシュされたsalienceでも、過去のcutに対して未来の支持を使えない。

Streamは直前のenergy scan一本と三つのscalar履歴だけを保持する。
scanは構築時に確保し、更新時は同じbufferを上書きする。同一の直近deliveryは再処理せず、
矛盾する再送、時刻の逆転、非finite値を拒否する。scan長はreleaseでもhard assertである。

## Frontendとの接続

入力はpowerとmono energyに加え、元のsource開始・終端・利用可能時刻を明示する。
NSGTの支持開始をcanonical hopの開始へ縮めない。availabilityはその入力が実際に利用可能になった
source clock上の時刻としてcallerが保証する。sliceの由来や実際の受信時刻をFrontend単独で証明はしない。

同じhopのenergy列、descriptor、accentは、lifecycle更新前のgroup handleを保持する。
更新前にeligibleだった親は成立hopを保持できるが、次hopのsuperseded親はassociationをmaskする。
新しい子groupは次hopから始まり、親の差分・accent履歴を継承しない。
energy snapshotにも元の列に対応するeligibilityを付ける。

観測欠測は全座標maskの区間とし、飛び越えた複数hopを一つの欠測recordへまとめる。
欠測後に観測が届けば、欠測recordと現在のraw descriptorを分けて返す。
既知の無音hopを補作せず、欠測をまたぐ差分を作らない。
正のmono energy・ゼロspectral massではscalar residualを保持する一方、
resolved groupのゼロ値を既知の無音としてdescriptorへ渡さない。

raw Streamの部分取得入力は取得sample数を保持し、全座標をmaskする。
部分区間の位置を使うDescriptorKnotへのclippingは未接続であり、このsample数だけで代替できない。
現在のFrontend入力は完全取得または欠測の二値である。

## 検査と残務

既存Python参照から24 sequence、1,152 endpointを生成し、Rustの値、mask、salience、採用判定、
原支持を照合する。合成scanで欠測、世代交換、実際のsplit、親のsupersession、子の初回差分、
未知shape、buffer再利用、入力失敗とsource cutを検査する。
構成PCMを実NSGTへ通す二bus検査でも、更新前handleと元のNSGT窓を保持する。
これは独立した音源回収、認知的妥当性、音楽的採用の検証ではない。

accent参照の全文SHAは登録値と一致する。descriptor参照の現全文SHAは過去の登録値と異なるが、
その登録SHAに一致する保存原本を特定し、今回使う`raw_descriptor`関数のsourceが完全一致すると確認した。
全文の一致へ読み替えず、両原本のSHAと関数照合結果を保存する。古い登録記録は上書きしない。

最終Rust検査は852成功・0失敗・22 ignored、追加分は10検査である。
Python参照の既存50検査も成功し、fixtureの再生成はbyte単位で一致した。
比較fixtureでは275 accentを採用した。実NSGTの構成PCM検査は二bus合計1,339 raw descriptorを作り、
accentは両busとも0であった。この実NSGT例だけで正のaccent検出の回収を証明したとはしない。
debug／releaseの同例の集計は一致した。

releaseの単一bus、各6,000更新の部分測定では、193 binsのp99は43.56 µs、769 binsは97.41 µsであった。
追加した前hop scanのpayloadは一Frontendにつきそれぞれ12,352／49,216 bytesである。
Frontend headerは82,184 bytes、返却Outputは89,888 bytesであり、これらは全heap／stack peakやRSSではない。
元のenergy scan、相関窓、Log2Space、allocator、実行環境の費用と区別する。

数値尺度は明示的なfixture入力であり、production向けにfit済みの尺度ではない。
部分的なrelease費用測定はgroup更新と今回の抽出を含むが、NSGT、全beam、queue、同時二bus／64 Voice、
実音声出力を含まない。全O04予算への合格とはしない。
accentのbounded delivery／累積credit、descriptorの圧縮・共有、beam／context所有、全fitと人のgateは残る。
