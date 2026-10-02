# I11-3 / T1：AMT 1.6.0適応C核の独立比較登録

2026-09-29。状態は**限定比較を単回実施し成立**。本書は[原典参照仕様](i11-t1-reference-spec-20260929.md)§4のAMT 1.6.0 C経路に限定した技術比較であり、Dau 1996原典再現ではない。Strube、整流、1 kHz/8 Hz LP、音圧校正、内部雑音、検出閾、音楽的費用、runtime配線、モデル採用は対象外である。登録と後の結果は本書へ集約し、結果を見て係数・許容・fixtureを選び直さない。

## 1. 比較対象と由来

| 項目 | 固定する内容 |
|---|---|
| 版と取得元 | [AMT公式配布案内](https://amtoolbox.org/download.php)から[1.6.0 source archive](https://sourceforge.net/projects/amtoolbox/files/AMT%201.x/amtoolbox-1.6.0.zip/download)。既存cache `target/i11-t1-primary-20260929/`を用い、再取得しない |
| 対象C | `amt160-source/amtoolbox-1.6.0/mex/comp_adaptloop.c`。SHA-256 `98d876031953b97c47b365b634710528123927a4834357b8ee89bfd949f8a854`。本登録時にarchive内の同名ファイルとのbyte一致を確認した |
| 原処理 | 同Cの `adaptloop_init` :21、`adaptloop_set` :41、制限なし `adaptloop_run` :112、`adaptloop_free` :32を呼ぶ。式をC側で書き直したものとの比較にしない |
| 由来 | C冒頭はStephan Ewert 1999–2004、Piotr MajdakによるAMT適応2016と記す。後年の実装であり、原著の実験プログラムという主張はしない |
| license記録 | archiveの`COPYING`は複数licenseであると明記する。隣接`mex/comp_adaptloop.m:17`はGPL v3以降を記すが、対象C自体にはlicense boilerplateがない。Cの個別許諾まで確定したとは書かず、元の著者表示、`COPYING`、`licenses/`を作業cacheに保持する。今回は既存公開sourceのローカル比較のみで、再配布・製品への取り込みを含めない |
| 比較名 | `AMT16-C-adaptloop-limit0-vs-independent-python-v1`。MEX、MATLAB fallback、Octaveを相互同値と扱わない |

以下のcache/source行番号は読取時点のもの。原論文とAMTの違いは参照仕様にまとめ、本書で新しい原典の主張を追加しない。

## 2. 原版Cを呼ぶ最小wrapper案

登録の作成単位では実装・compileを行わなかった。後続単位の作業物は `target/i11-t1-adaptation-20260929/` に閉じる案とし、mainの`src/`、Cargo設定、既存worktreeを変更しない。

1. 上記hashのCを変更せずincludeする小さなC translation unitを作る。`mex.h`だけをローカルshimにする。型`mxArray`と`mxREAL`、原版で参照される8関数（`mexErrMsgTxt`、`mxGetM`、`mxGetN`、`mxIsDouble`、`mxIsComplex`、`mxGetScalar`、`mxGetPr`、`mxCreateDoubleMatrix`）を宣言・定義し、**全stubは呼ばれたら即abort**する。MEX gatewayも元ファイル内に残すが呼ばない。これはMEX互換実装の試験ではない。
2. wrapperは1 channel・5 loopsのcreate/run/snapshot/restore/freeだけを外へ出す。createは原版init/set、runは原版run、freeは原版freeを呼ぶ。snapshotは5個の`state`をコピーする。restoreは同じ固定パラメータでcreateした新stateへ5個をコピーする。構造体のpointerを浅くコピーしない。
3. `a1`、`corr`、`mult`、`limit`、`minlvl`、`loops`、`nsigs`は設定後の不変値として読み取り記録する。factor/expfac/offsetも原版setに計算させるが、limit=0経路では読まない。wrapperはMU変換、floor、再帰、係数を代行しない。
4. compile案はC11、`-O0 -fPIC -shared -fno-fast-math -ffp-contract=off`、標準数学library `-lm`。通常のbinary64、既定round-to-nearestを要求し、Cのfenv照合を実行前後に記録する。compiler/version、全argv、platform、C・shim・wrapper・Pythonのhashを記録する。手元compilerが条件を満たせない場合は未実施として戻す。最適化や性能を比較しない。
5. Python標準libraryの`ctypes`から呼ぶ。外部packageのinstall、MATLAB/Octaveのinstallは不要。wrapperの事前検証で不正値を原版Cへ渡さない。 allocation失敗・stub到達・非finite出力はその時点で失敗とし、代替実装へ自動で切り替えない。

独立参照はPythonの通常のfloat、`math.exp`、`math.pow`、明示的な5段ループで新規に書く。AMTのMATLABコードやC再帰を文字列変換せず、下の式から実装する。共有するものはfixture、固定パラメータ、binary64の入出力、比較器だけ。Cの状態・係数をPython再帰へ流用しない。OSの数学libraryを共有し得るので、超越関数まで独立したoracleとは呼ばない。代数的な定常検査を別に置く。

## 3. 固定パラメータ・式・時計

| 項目 | 登録値と理由 |
|---|---|
| channel / loops / limit | 1 / 5 / 0。AMT16 `defaults/arg_adaptloop.m:24`の`adt_dau1996`選択に対応し、単独関数のlimit=10を使わない |
| 時定数τ | `[0.005, 0.050, 0.129, 0.253, 0.500]`秒。AMT16 Cの参照条件であり、Conchordalの採用値ではない |
| sample clock | `fs=48000`整数、標本nは`n/fs`秒、入力区間は`[0,N)`。全実装・chunk・forkで同じ標本列を使う。このrateをDau原実装の時計と呼ばない |
| floor / 尺度 | `minspl=0`、内部の振幅1＝100 dBというAMT規約から `m=10^((0−100)/20)=1e−5`。入力はこの規約の線形振幅。実PCMからdB SPLへの校正は行わない |
| 初期状態 | 原版Cはsqrtを5回適用。独立参照は `qj(0)=m^(2^(−j))`、j=1..5を個別powで求める。初期値の差も比較対象に残す |
| 係数・MU | `aj=exp(−1/(τj fs))`、`c=m^(1/32)`、`k=100/(1−c)`。PythonはCの値を読み戻して使わない |

各標本で `y0=max(xn,m)`、段jを1から5へ進め、`yj=y(j−1)/qj_old`、続いて `qj_new=aj qj_old+(1−aj)yj`。出力は `u=(y5−c)k`。除数は更新前の値である。状態qは正の有限値を要求する。MUを0..100へclampせず、段差のovershootや切断後の負値が出ても、それだけで失敗としない。

## 4. 有限fixtureと比較の分母

全数値は十進literalからbinary64へ一度変換する。入力配列は1回作り、byte単位で同じ配列を両実装へ渡す。乱数、音声file、探索は使わない。nの範囲はすべて左閉右開。

| ID | 入力・初期状態・長さ | 検査と分母 |
|---|---|---|
| Z | `x=0`、通常floor初期化、N=256 | C対Pythonの256出力・最終5状態。出力0、状態`m^(2^(−j))`との代数比較も同じ許容で行う |
| F | `x=m`、通常初期化、N=256 | 同上。ZとFのC出力・最終状態はbit一致を要求する。floor以下とfloor値が同じ経路へ入ることの検査 |
| B | `[0,−m,m/2,m]`を32回、N=128、通常初期化 | C対Python128出力・最終5状態。Cは同長のfloor列とbit一致。有限の負入力は原Cのfloor動作を確認するための範囲外刺激であり、不正値としてrejectしない |
| S | `x=v=1e−3`、N=512。両実装の状態を解析的定常値`qj=v^(2^(−j))`へ独立にrestoreして開始 | 512出力・最終5状態を比較。解析MU `100(v^(1/32)−m^(1/32))/(1−m^(1/32))`と定常状態を別検査。これは任意の待ち時間で収束したと仮定する試験ではない |
| D | N=62400。`[0,4800)`は0、`[4800,24000)`は1e−3、`[24000,62400)`は0。通常初期化 | C対Python62400出力、最終5状態、および下記10境界の5状態。ゼロ、立上り、切断、回復を1本で比較。単調回復や指定秒以内の完全復帰を追加条件にしない |
| FA / FB | Dのprefix `[0,24000)`を処理したcutから各2048標本。FAは全0、FBは先頭512標本だけ1e−3、残り0 | 各2048出力・最終5状態をC対Python比較。各々を通常初期化からprefix＋branchで処理した出力suffix・最終状態とも比較する |

一次C対Pythonの出力分母は **67648標本（7 fixture）**、最終状態は **35要素**。Dの境界状態は別欄で50要素とし、最終境界を含む重複を隠して分母へ足し込まない。snapshotによりrunを分割する経路と全長runを別に作り、全長側のstateを途中で置き換えない。

Dのchunk境界は `[0,1,17,4096,4800,4801,24000,24001,28096,62400]` に固定する。短いchunk、段差の直前直後、切断の直前直後を含める。C内の全長対chunk、Python内の全長対chunkは、それぞれ62400出力と最終5状態のbit一致を要求する。境界4800で空chunkを1回挿入し、wrapperのno-opとして状態byte不変を確認する。空chunkでは原版runを呼ばない。

forkではC側とPython側がそれぞれ自分のcutを保存する。FA→FBとFB→FAの両順を実行し、順序によらず各branchの出力・終状態がbit一致すること、母cutの5状態と不変係数がbyte不変であることを確認する。さらに母状態からDの本来のsuffixを続け、branchなしのDとbit一致を要求する。branch評価結果を母状態へcommitしない。FAとFBの出力には許容を超える差が少なくとも1標本あることを要求し、両者が同じ配列を誤って参照した場合を検出する。

## 5. 許容、入力境界、判定

| 比較 | 登録する規則・数値の意味 |
|---|---|
| C対Python、代数MU | 各標本で `abs(C−R) ≤ 1e−9 + 1e−11 abs(R)`、Rは独立参照または解析値。単位はMU。0近傍の相対誤差を分母にしない |
| C対Python、代数状態・係数 | `abs(C−R) ≤ 1e−12 + 1e−11 abs(R)`。状態、a、c、kを別欄に記録し、異なる単位の最大値をまとめない |
| 同一実装内chunk/fork | binary64のbyte一致。処理順を変えず状態だけを保存するため、許容誤差で履歴欠落を救済しない |
| 数値根拠 | binary64のepsilonは約2.22e−16。相対1e−11はepsilon約45000倍で、sqrt対pow・libm・62400段の丸め差を許す事前の工学的比較幅である。安定性の数学的誤差上界や知覚閾を保証しない。MUの絶対1e−9は0近傍、状態の1e−12は小さい除数近傍を比較可能にする。結果から幅を拡張しない |

wrapperは固定値以外のfs、limit、m、τ、channel/loop数をrejectする。入力列は有限binary64、長さは非負整数かつこの登録の最大62400以下、restore状態は長さ5・全て正の有限値を要求する。入力buffer長不一致をCへ渡さない。`m>0`、`m<1`、τ>0を満たし、MU分母`1−c`と各qが正の有限値であることをcreate/restore/run前後に確認する。途中の非finite出力や最終不正状態を平均値に埋め込まない。

不正入力の最小fixtureは10件：`fs=0`、`fs=48000.5`、`limit=10`、`m=0`、`m=1`、先頭`τ=0`、入力1個をNaN、入力1個を+Inf、restore先頭q=0、restore長4。すべて「C核を呼ぶ前にreject、出力なし、既存状態byte不変」を期待する。その他の全組合せや任意の異常値を網羅する試験は増やさない。

結果は各fixtureについてN、比較要素数、違反数、最大絶対誤差とそのindex、最大`abs(C−R)/(atol+rtol abs(R))`、最初の違反のC/R値を記録する。非finite数は独立欄にする。bit比較、fork母状態、10件のrejectはそれぞれ件数と期待結果を記録し、欠測をpass分母から除いて見かけの合格率を作らない。

## 6. 終了点と後続結果欄

後続実施の入口は、統括の登録レビュー完了、元C/archiveのhash一致、実装物の静的レビュー、既存のtiming lockに反しない実行枠の確認とする。新しい性能測定や資源gateは作らない。実施は上記fixtureの1回取得で閉じる。失敗時は原版hash・入力・wrapper境界・最初の差を報告し、登録変更や自動再取得は行わない。

全比較成立時の結論は「指定したAMT16 C核と独立再帰が、この有限条件と許容内で一致し、保持・分岐のwrapper契約を満たした」に限定する。Dau原典のgolden出力、聴覚妥当性、sample-rate一般性、f32精度、複数channel、runtime性能、T1入力適合、モデル採用を証明しない。

| 後続記録 | 状態 |
|---|---|
| 統括レビュー・登録版hash | 2026-09-29：統括が原Cのinit/set/run、状態更新順、7 fixtureの67648標本、35最終状態、50境界状態、比較範囲と事前許容を確認。限定比較の実装へ進める。実行版は静的レビュー後にhash固定する |
| wrapper/参照実装と静的レビュー | 原Cを無変更includeするC wrapper、abortするMEX shim、式から実装したPython参照と比較器を作成。統括が呼出し・更新順・fixture・chunk/fork・拒否を静的確認 |
| compiler・実行環境・command・artifact hash | [manifest](../../../target/i11-t1-adaptation-20260929/acquisition/manifest.json)。cc Ubuntu 15.2.0、登録flags、fenv開始/終了ともFE_TONEAREST。凍結登録SHA `06e81e9f1bd3c67623bc488879e97a0da0106620f4a7be6f58c4d82033843731` |
| 7 fixture、chunk、fork、10 rejectの結果 | 一次67648出力・35最終状態、別集計のD境界50状態。47比較の違反0・非有限0。chunk/fork等のbit検査8記録、10件の不正入力拒否が成立 |
| 失敗または成立範囲・次の作者判断 | 指定AMT16 C核と独立再帰の有限比較が成立。全Dau、知覚妥当性、f32、runtime、T1地形、モデル採用は未判定 |

登録作成時はC/cacheの読取、hash一致確認、参照確認だけを行った。後続実施を以下に記録する。main sourceは変更していない。

## 7. 単回比較の結果

2026-09-29 15:32:37 JST、[実行終了記録](../../../target/i11-t1-adaptation-20260929/acquisition-status.txt)はexit 0。[結果JSON](../../../target/i11-t1-adaptation-20260929/acquisition/result.json)、[凍結登録](../../../target/i11-t1-adaptation-20260929/acquisition/registration-frozen.md)、21本のinput/C出力/Python出力binary64列とhashを保存した。係数・fixture・許容を変えず、一度の取得で終了した。

出力の最大絶対差はDの **3.637978807091713e−12 MU**、許容で割った最大差は約8.3544e−5だった。これはCとPythonがbit一致したという意味ではない。Z/Fの代数ゼロとの差は約3.6742e−14 MUであり、登録許容内の丸め差として保持する。同一実装内のchunk分割・候補branch順・母状態継続は登録どおりbit一致した。

統括は比較器のpass欄だけでなく、保存した67648対の出力byte列を別の読取処理で再計算し、全件の有限性・登録許容・最大絶対差を確認した。凍結登録・使用sourceと21保存列のhashも照合した。状態・係数・拒否検査は静的レビュー済み比較器の保存記録を確認した範囲であり、別実装による全47欄の再実行ではない。

これで状態保持・候補分岐を持つ後年適応核の最小参照が得られた。次はモデル路線とPCM入力契約の具体案を作者判断へ渡す。未来背景energyから原典波形を作ったことにはならず、原Dauの未入手設定を本結果で埋めない。
