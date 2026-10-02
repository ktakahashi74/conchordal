# F4 固定二身体比較の移動停止理由: 計測前登録

2026-09-27。対象は[四条件比較の取得前登録](../../.worktrees/body-fitness-factorial/docs/design-notes/body-fitness-factorial-registration-20260927.md)と[取得結果](../../.worktrees/body-fitness-factorial/docs/design-notes/body-fitness-factorial-results-20260927.md)で、prepared判断が各声11回消費されたにもかかわらず、四条件ともtargetと実f0が初期値から動かなかった理由である。本登録は同じ判断の数値診断に限る。移動を起こす条件の探索やF4全体の採否判定ではない。

## 固定入力と取得順序

既存のRhai scene、四条件（`point_point`、`body_point`、`point_body`、`body_body`）、seed 7、音源と初期f0の割当、gain、既定を含む全制御係数、48,000 Hz、512 samples/hop、Finish 0.64 sを変更しない。各条件を従来どおり2回、通常の`conchordal-render`で取得する。診断用reportは判断の観測だけに使い、採点値、候補生成、RNG、音声、代謝、targetの更新に関与させない。

診断実装の前に、四条件比較の旧ソース一式（未コミット差分を含む）、sceneとconfig、旧raw report、WAV、比較summary、検証ログを読み取り専用の固定先へ複製し、各ファイルのSHA-256と取得時点をmanifestへ記録する。旧結果を上書きしない。固定の完了とhash照合を確認してから診断コードを変更し、新版を別パスへ取得する。新版でもソース一式・入力・raw・WAV・検証ログ・manifestを別に固定する。旧版の既報WAV SHA-256は`222edba6373a4f8bc5740db6d5e0458e6ca308699d0ccc90b876965cecefd322`であり、実際の固定ファイルから再照合する。

## 実消費計算から記録する量

成功として消費された各prepared判断について、source ID・generation・birth sample、条件、判断時計、support終端、解析epoch、space/habituation版、実VoiceのRecipe identity、target・現f0、候補pitch bitsとprepared raw scoreを記録する。拒否は理由と時計を別に記録し、成功判断の採点値に混ぜない。RNG stateは判定時の既存の完全照合を維持し、診断reportのために追加サンプリングしない。

`PitchHillClimbPitchCore::propose_with_prepared_body_scores`から`propose_with_scorer`へ渡した**その回の実際の採点**を記録する。候補選択後のbestと現target baselineの両方について、採点に使ったpitch、prepared raw score、landscape weightと目的符号適用後の加重score、現実f0からの距離、move-cost時間尺度・時間・係数・指数と減点値、tessitura center・gravityと減点値、crowding入力と減点値、adaptationの参照indexと加点値、最終adjusted scoreを出す。`best_score - current_adjusted`、閾値`0.1`、温度、閾値分岐またはMetropolis分岐、分岐直後のproposal、controllerでの範囲clamp後のtarget、音高適用後の実f0も記録する。bestの同点選択がある場合は選択に使った距離とRNG分岐の有無を示す。

内訳は実消費で評価済みの値をその場でreport用データへ写す。診断目的の二度目のscorer呼出し、候補再生成、追加RNG呼出し、別のlandscape評価をhop経路へ入れない。`score_uses`は採点関数の使用回数であり、target採択回数とは分ける。初回だけでなく全成功判断を記録し、「一度も動かなかった」60 frame全体を説明する。候補ごとの全内訳を保存する必要はないが、bestが選ばれるまでの候補pitch・adjusted scoreと同点規則を照合できる最小記録を残す。

## 独立照合と判定

新版rawから独立した検査器を作り、本体の`adjusted_pitch_score_impl`や`propose_with_scorer`を呼ばずに計算する。入力は保存したsource-removed eff scan、Log2Space、代表身体Recipeと密度、prepared候補score、制御値、採点時のoccupancy/adaptation値、および実消費記録とする。点scoreは同一scanのlog2補間、身体scoreは同一scanと代表身体密度の積分として別に計算する。bestとbaselineについて加重score・各減点/加点・adjusted score・改善幅を式どおり再構成し、候補選択、閾値分岐、proposal、clamp後targetを再判定する。検査器の算術順序とf32丸め差を明示し、各量の誤差上限を記録する。数値許容差は絶対`1e-5`と相対`1e-5`の大きい方を上限とし、閾値との差がその上限以下なら採択理由を未解決とする。rawから再構成できない内部状態があれば、その項目は独立照合済みと呼ばない。

診断の主判定は、各成功判断で次を区別する。(1)候補が存在しraw scoreも消費されたが、調整済みbestとbaselineの差が閾値以下で、温度0のためproposalが現targetのまま。(2)閾値を超えてproposalは変わったが、controllerの範囲clampで戻った。(3)targetは変わったが、Glide等の音高適用段で実f0が変わらなかった。(4)準備拒否・欠測・検査不一致によって原因を決められない。実測の内訳が示した項目だけを原因とし、move costや閾値を事前の推測だけで確定しない。四条件間で点/身体score差を保ったまま各判断の分類を示し、診断できた回数と未解決回数を併記する。

同一条件の2回で決定的なreport項目、判断回数と拒否理由、target・現f0系列、WAV hashが一致することを確認する。旧版対新版では同じscene/config hashに加え、旧rawと新版rawの共通フィールド（source identity、時計、候補、prepared score、RNG probe、判断結果、代謝、target・現f0）とWAV hashを比較する。診断追加だけでこれらが変化した場合はreport-onlyの不変性が未成立として記録し、旧版の理由を新版から遡及して断定しない。完全RNG照合は実消費時の既存検証であり、一値probeだけを完全一致の証明には使わない。

本単位では身体と初期f0の交換、温度・move cost・proposal interval等の変更、期間延長、seed探索を行わない。それらは音色と位置の交絡解消や実移動の検証に必要なら、別の取得前登録で入力と成功条件を固定する。
