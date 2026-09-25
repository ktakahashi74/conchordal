# I11-2: 到来CDFを発音候補の比較へ返す

2026-09-24。実装前登録。Astraの独立レビュー「可」を受け、
[入力・実行順](i11-2-inputs/plan.json)と[凍結記録](i11-2-inputs/freeze.json)を保存した。
凍結時点では第2段の実装・結果取得・完了判定を行っていない。

本書は[I11登録の§4.5・§5・§6](i11-onset-comparison.md)を第2段について具体化する。
第1段の限定完了、作者採用（A1／A2）、資源受入（R2）、O15／O18の達成とは分ける。
第2段も既定は`None`のままにする。研究用の係数を、学習済み・適合済みの係数と呼ばない。
結果に応じた素材・係数・許容の変更は本登録の成功に数えず、新しい登録として残す。

## 1. 比較量と数値計算

CDFは発行時刻を起点とする次の実accentの到来確率である。時刻はsample clock、
格子は`k = 0..200`、`t_k = (k as f64) / 50.0`秒。sample rateを正の整数`fs`とし、
格子の相対sample座標は`4 * fs * k / 200`を`f64`で計算する。格子時刻を整数へ丸めない。
`horizon_end = issued_at + 4 * fs`は整数で、201点目と一致する。時刻加算がoverflowする場合は表を作らない。
本取得の`fs = 48,000`では960 sample刻み、地平は192,000 sampleである。
設定の`horizon_sec`が4.0でない場合は`unsupported_horizon`とし、別の地平へ暗黙に伸ばさない。

主Hazard設定を次に固定する。18成分の意味・softplus・標準化はI6の実装をそのまま使う。

```toml
[temporal_period]
model = "hazard"
coefficients = [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
means = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
deviations = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
horizon_sec = 4.0
```

これは位相cos成分（index 10）の作用を通す診断値である。定数2/秒のHazard
（切片`ln(expm1(2))`、他0）を解析対照、`model = "periodic"`を段差の対照にする。
定数切片の実際の`f64`値・bit、全設定・シナリオのbytesとSHA-256は入力manifestへ固定する。

### 1.1 既存積分からの変更範囲

既存`Engine::integral`は二点Gauss–Legendre、panel数1・2・4・8・16の有界計算である。
位相係数が非0のときはpanel幅が周期の1/4以下という条件もあるため、周期が1秒未満の
音群に対する全4秒の積分は、この上限では成功しない。これは結果取得前のソース検査で判明した。

第2段専用APIは、確定した経過時間`e`について各隣接区間`[e+t_(k-1), e+t_k]`を
既存`integral`で積分する。昇順に`f64`で累積し、`F[0] = 0`、`F[k] = -expm1(-sum_k)`とする。
1区間でも未収束・非有限なら表全体を不明にし、部分表や0の表を配送しない。
区間当たり最大62回、1音群当たり12,400回、7音群で86,800回のhazard評価が上限となる。
生成数、評価数、失敗数、生成時間、observer全体の処理・配送時間を報告する。

既存I6の`probability`、`integral`、履歴更新は変更しない。全4秒積分に由来する既存
`Forecast.probability = None`を、新CDFの支持不足へ読み替えない。新CDFは確定elapsed、
reset状態、contextの因果性、各小区間の収束で独立に判定する。I6の単一地平確率と
第2段CDFの支持状態が違う場合も、両方を報告して追跡可能にする。

小区間の局所収束は、累積全体の精度の証明ではない。独立Python参照はEngineの積分コードを
移植せず、適応Simpson積分で累積hazardを求める。初期区間幅を20 ms以下、位相項がある場合は
さらに周期の1/8以下とし、各20 ms格子区間を等分してこの上限を満たす。
4秒全体の絶対許容`1e-11`を初期区間の長さに比例して配分し、再帰二分時も半分ずつ配分する。
再帰深度上限24で未収束なら参照失敗とする。周期の整数倍の疎な標本だけを使うと
粗細の差が0になり得るため、全4秒を一つの初期区間から積分しない。
201点すべてで累積hazardの差を
`1e-8 + 1e-6 * abs(I_ref)`以下、CDFの差も同じ上限以下とする。
既存全区間積分が成功する格子点では、それとも同じ許容で照合する。
定数Hazardの解析式、非定数・位相依存fixture、Periodicの段差を別々に検査する。

`elapsed_seconds[0] != elapsed_seconds[1]`または`reset_unknown`なら表は不明である。
Periodicは次の予測到来で0から1になる段階関数を格子点へ置く。格子間の消費はHazardと同じ
線形補間とし、段差を補間する幅も20 msと明記する。周期peakのない音群は消費集合から外す。

## 2. 配送と識別

観測workerが不変の`ArrivalPublication`を作り、既存の`observation::Snapshot`に
`Option<Arc<...>>`として載せる。各表はmodel/version、group handle（bus・epoch・世代を含む）、
source support、available、issued、horizon、periodic flag、elapsed/reset状態、表の値または
不明理由を持つ。7音群の固定容量とし、Voiceごとの表の複製をしない。
Engine、係数、履歴、context本体はVoiceへ配送しない。数値核の独立検査用fixtureは別artifactとする。

CDF・period・action-profile対応表を同じ観測状態から作り、同じsnapshot lockで一体に公開する。
runtimeはhopごとに一つのsnapshotを取り、そのepoch・対応表・CDFを同時に保持する。
同じ共有snapshotをUI／reportも読む。取得失敗で前回値を保持しても、consumerで期限と識別を検査する。
新epoch、音群退役、worker失敗で現行の表を消す。旧Arcが残っていても新世代へ付け替えない。
CDFの生成は`arrival=true`のhabitat bus 0だけで行う。bus 1、通常の`None`、
`arrival=false`では生成しない。上記86,800回はこの1 bus・1発行の上限である。

受領時刻はruntimeがその発行版を初めて取得したsample clockとし、同じ版を保持するhopで更新しない。
`source_end <= available <= issued_at <= received_at <= now < horizon_end`を検査する。
現在の通常観測は`available = end = issued_at`である。壁時計の処理時間はこの順序とは別に測る。

自己除外にはI10のBindingと同じsnapshotのaction-profile Keyを使う。
source id・source generation・body generation・bus・body model versionを現在のVoiceと照合し、
profile SHAと公開Keyの一致、`end <= available <= now`、`now - end < 24,000`、
`assignment_end <= issued_at <= now`、`now - assignment_end < 24,000`、
対応表の`issued_at <= arrival.issued_at`、prototypeの範囲・routed指定を検査する。`groups[prototype]`のbus・epoch・現行世代が一致した
handleだけを除外する。24,000 sampleは既存consumerの期限であり、秒への読み替えはしない。
複数の有効Bindingがあれば対応するhandleをすべて除外する。無効・欠落の場合は推測せず、
自己除外できなかった理由を報告する。記述子対応による関連付けであり、完全な自声分離は主張しない。

## 3. 候補評価と報告

元の23候補格子、earliestによる除外、T1の同一record・D・16 bin、外部予測を維持する。
到来集合は実際に評価される候補（earliest除外後）の全域に対して一度だけ確定する。
bus 0、現epoch・世代、有効時計、periodic、表あり、自己以外という条件に加え、どれか一つの
候補で`at + width > horizon_end`ならその音群を全候補から除外する。
下端は`max(at-width, issued_at)`へ切り詰める。上端が地平と等しい場合は使用できる。
同じ音群の複数版がある場合は、現snapshotの最新発行版を一つだけ使う。

`P_g = F_g(at+width) - F_g(max(at-width, issued_at))`を線形補間し、handleのslot順に
`f64`で足して音群数で割る。空集合ならunknownで、既知の`P = 0`とは区別する。
補間位置は発行時刻からのsample差を`fs / 50.0`で割った`q`とし、`i = floor(q)`で
`F[i] + (F[i+1] - F[i]) * (q-i)`を使う。起点以前は`F[0]`、地平端点は`F[200]`。
候補窓の上端が地平を超える表はこの補間へ入る前に除外する。明示的な融合積和は使わない。
費用項は`(f64(coupling) * arrival_weight * (1 - P)) as f32`を一度だけ丸め、
既存の`f32` base費用へ加算する。既存項の演算順序は変えない。
`arrival=false`、集合なしでは加算自体を行わない。候補同点は既存の格子順を維持する。

選択は到来込み費用で行う。skipは**選ばれた同じ候補の到来項を加える前のbase費用**で判定する。
浮動小数の引き算でbaseを復元せず、選択時に保存する。base費用が最小の別候補へ置き換えない。
選択候補が変わればbase費用も変わり、skipだけが最初に分岐する場合もある。
その場合も独立再計算と一致すれば正当な作用である。予約後の新着CDFで予約を変更しない。

reportには受領版と時計、使った表、音群別の採否・理由、自己除外の照合情報、各候補のPと
到来費用・base費用・総費用、選択・skipを残す。表なし、未来、期限切れ、地平不足、非周期、
経過時間不確実、reset、積分失敗、異bus／epoch／世代、自己一致を区別する。
`arrival=false`、weight 0、coupling 0、既知CDF 0、全群unknownを別の対照とする。
到来項を新しい学習の教師・信用として使わない。
独立参照は新しいPと到来項の各`f64`演算、最後の`f32` cast、`f32`のbaseへの加算を再現し、
bitで照合する。設定の`arrival_weight`は既存の`f64`型を維持する。
既存base各項の独立近似再計算には第1段の`1e-7 + 1e-5 * max(abs(a), abs(b))`を引き継ぐが、
介入間のbase・入力の比較や新到来項の照合へこの許容を流用しない。

## 4. 取得行列と合格条件

入力は第1段の登録12シナリオを全件保持し、sine／harmonic／modal × 総4／16 Voiceの
外部pulse付き6条件を追加する。新6条件は1声の2 Hz外部pulseと3／15声の選択対象からなる
明示的な同期assayである。seed 20260924、48 kHz、6秒本体＋0.5秒終端。
具体的Rhai、config、plan、hashは[i11-2-inputs](i11-2-inputs/README.md)に添付する。
結果を見て素材を差し替えない。
新6条件ではpulseをhabitatのみ、対象Voiceをpresentationのみへ送る。これは自己混入をbusで
分けた診断条件で、混合音からの完全な自己分離を示すものではない。§5.9は元の登録どおり、
発行されたToneEnergyを規約上bus 0へ単独投影し、身体のshapeを比較する。
実際のrouteで投影を消音しない。旧12条件と新6条件を別母集団として集計し、実routeを併記する。

| 検査 | 第2段の必須条件 |
|---|---|
| §5.1・5.2 | 全使用入力とreportが一致。独立参照でCDF補間・各費用・選択・skipを再計算。同じ凍結入力に対するbody/proxyとarrival on/offの介入を別々に検査する。 |
| §5.3・5.6 | 未来受領、期限、支持外、全候補共通集合、欠測と復帰、bus／epoch／世代／モデル交換、Binding欠落、旧返却・drop再送、予約後非再選択を強制fixtureで検査する。 |
| §5.4a | 元タグのcommit `103a32fe5c5fc4fdf0f4d65fb25a1bb0c7c47303` と最終版のNoneを同じMac・同じ入力で比較。旧12条件のWAV SHAと学習10種がbit一致。除外欄を列挙し、新機能のoffに不要な除外は増やさない。 |
| §5.4b | 改訂済み登録どおり共通candidate recordの決定的内容を比較。片側のみ・件数差・dropは別集計。凍結packet再評価は未実施という限界を保持し、共通record比較をその代用証明と呼ばない。 |
| §5.5 T1 | 最終版arrival=falseの旧12条件で、最初の選択分岐までbody/proxyのrecord・D・外部入力が同一、違いがpowerだけであることを検査。第1段で作用を示したharmonic／modal各4／16／64 Voiceの6条件すべてで選択・WAV差を必須とし、sineの差なしも全件報告する。 |
| §5.5 T2 | 新6条件のbody arrival=false/trueを比較。最初の選択・skip分岐まで、全非到来入力と候補時刻をbit比較。on側CDFを用いた同一入力への独立介入で、差が到来項から生じることを確認。3身体それぞれ少なくとも1条件で選択またはskipとWAVが変わること。 |
| §5.7 | 下記の全hop負荷の主判定を全組で満たす。 |
| §5.8 | 201点の累積hazard・CDF独立照合、解析Hazard、Periodic、全不明理由と集合固定を検査する。 |
| §5.9 | arrival=trueの旧12＋新6条件で実際のonsetに対応するToneEnergyを再投影。bodyとproxyを別母集団に分け、power／overlap／term gapの中央値・p95・最大と除外件数を報告。差の大小に合格閾値は置かない。 |

実走offではCDFを作らないので、on/off間の「同じCDFを配送した」という主張はしない。
同じCDF上の介入は凍結入力検査で担い、実走では非到来入力の同一prefixとon側入力の独立再計算で
原因を照合する。到来以外の差が先行した比較は不合格。作用なし・決定0・適格群なしを成功に数えない。
`Some`対`None`は分割数と打切りを含む参考比較として別に報告する。

### 4.1 全hop負荷の事前規則

対象は旧12条件のうち4／16 Voiceの8条件と新pulse6条件、report有／無の28組。
新6条件のon側では、report有／無それぞれの各反復で有効な到来項の消費が1件以上あることを
適格条件とする。reportなしでもprofileの使用件数で確認する。
生成・消費がない軽い経路の負荷を、T2が働く経路の合格へ数えない。
`population_us`、`synthesis_us`、総`elapsed_us`の中央値・p99・最大、予算超過hop数の10欄を使う。
待ちを引いた`own_us`へ合格対象を置き換えない。

同じ機械・同じ第2段binary・同じ主Hazard設定のNone/Noneを交互3反復して各組を測る。
これを3 pass（84組）行う。集約・測定区間は第1段の`hop_path.py`と同じ規則で固定し、
各欄のfloorは84組のA/A絶対差の最大値とする。pass 1〜2から得たfloorでpass 3を判定した
超過数も報告するが、その結果でpass数やfloor算出式を変えない。

主判定は中央値・p99について`abs(new-base) <= max(0.05*base, floor)`、最大について
`abs(new-base) <= max(1,000 us, floor)`、超過hop数について`abs(new-base) <= floor`とする。
各runの全記録hopを使い、中央値・p99は整列値の`floor(q*(N-1))`番目とする。
9個の時間統計はrunごとに求め、その各統計の3反復間中央値を比較する。
主判定の超過hop数は3反復の合計であり、floorもその合計差から得る。
floorを保存・hash固定してから候補比較を取得する。元の±5%／1 ms／超過同数の判定も併記する。
元判定の超過同数は`hop_path.py`どおり3反復の件数の整列リスト一致を使い、合計一致と混同しない。
floor適合を統計的同等性の証明とは呼ばない。

比較は最終版の`body arrival=true`対`None`、および`body arrival=true`対`body arrival=false`。
両比較とも28組・10欄の全件合格を必須とし、同じNone A/A floorを用いる。
各組は3反復を交互取得し、反復・passで先行変種を交替する。全runの順序はplanへ固定する。
instrument、render、長いbuildは並列実行しない。終了コードとartifact hashをrunごとに保存する。
CDF生成のobserver負荷、footprint返却遅延、absent比率、置換・dropと待ち時間も別表で報告する。

## 5. 基準の保存、凍結、完了宣言

第2段着手前の`7f76e0b970b1ed06ebce1a60e9cee71f35ecbb8f`の実行binaryを
`target/i11-stage2-20260924/baseline/`へ保存した。旧buildのsource SHAと全209ソースfileを照合し、
元build証跡も保存した。現作業treeの後日の変更をこのbinaryの出所と呼ばない。
render SHAは`29f15c2af1246601164d1769790fe1860db341ae544671c4ac533862b667a688`、
instrument SHAは`eb385ee8d4eba4ca5e660b47aa4343ab3159d4bb88777c3c9ebc90bb08615dde`。
元None基準タグから同じMacでbuildしたbinaryは別に保存済みで、両基準を混同しない。

入力・判定規則・baselineのmanifestを完成させ、[Astraの独立レビュー](i11-2-inputs/prereg-review.md)後に
hashと日時を凍結記録へ保存した。候補binary・実装source・検査器の版は実装後かつ取得前に
別の実行manifestへ固定する。供給・配送・消費と独立検査器を実装し、その後に取得を行う。
ソース変更後の必須cargo test（完全出力・終了コード）、fmt、clippyを通し、結果と限界を記録して
Astraへ最終レビューを依頼する。全必須条件とレビューを満たすまで第2段の技術完了を宣言しない。
