# F4b: 出生一機会への接続案

日付: 2026-09-26。状態: 静的調査による実装案。取得登録・実装済み・出生効果の実証ではない。
関連: [F4消費境界](body-aware-fitness-f4-consumer-audit.md)、[F4a成人代謝](body-aware-fitness-f4a-metabolism-registration.md)。

## 最初に通す入口

初版は実`Community::respawn_on_new_deaths`の一機会、`RespawnPolicy::Random`、
周波数範囲内の一様候補、`respawn_settle_strategy=None`、固定ratioのHarmonic身体に限る。
旧コードの16候補、正のscoreによる重み付き選択、全重みゼロ時の最大score選択、
採用後のmin level判定を維持する。候補抽出・親・身体・目的関数を一度に変えない。
Hereditary、PeakBiased、初期Population配置、音色遺伝、長期選択は別の取得単位とする。

新生児には自己音がまだない。評価環境は出生前の共有解析とし、死亡した旧Voiceの残響も
他の音として残す。F3の成人LOOを、死亡個体や選ばれた親のidで流用しない。
固定された共有環境と子の各候補の実ToneからF2のscoreとlevelを求め、
候補の重みと出生閾値に同じ評価を使う。

## 同一性と乱数

最小試験は明示offlineで実行する。試験台帳で次の未使用runtime idを指定し、
候補採点時のidと採用後の実`allocate_runtime_id`の値が一致することを要求する。
Communityのid消費順序は変えない。拒否された機会ではspawn counterだけが一回進み、
idとmember indexは進まないという旧規則を対照と比較する。非同期予約への一般化は含まない。

候補ごとに、実生成と同じtemplate、候補基音、予約id、開始frame、metadata、
Community seed、出生前地形から試験用Voiceを構築する。そこから代表Tone recipeを得る。
候補生成・選択の乱数をこの構築へ渡さず、`spawn_with_landscape`自身の乱数初期化を使う。
採用後の実Voiceについて、id、generation、BodySnapshot、recipe、現在基音を照合する。
ModePatternが出生地形や基音に依存する場合は別登録とし、この固定身体試験で保証しない。

## 検査の最小構成

1. 実死亡検出から出生までを通し、16候補、全score/level、候補選択前後の乱数、
   採用候補、子のidと身体、runtime eventを保存する。参照側は同じ候補と選択前乱数から
   F2の表を使って選択を独立に再計算する。
2. 出生前の共有環境には旧Voiceの残響を含める。旧Voiceを除いた成人LOOと異なる入力であると
   PCMと密度で確認する。親や死亡個体の評価を子の評価として使わない。
3. 身体評価用の封印環境を維持してCommunityへ渡す旧点Cだけを変更し、
   身体方式の選択、閾値判定、乱数列が同じであることを確認する。
   `None`方式には旧点Cによる結果を残す。候補生成は一様に限定し、旧Cへの依存を混ぜない。
4. 閾値なし、全候補の身体levelより低い閾値、全候補より高い閾値を、取得前に定めた規則で
   独立した同一初期状態へ与える。成功と拒否の両方でcounter、id、member index、eventを検査する。
5. template、予約id、出生時刻、環境版、候補keyの不一致では出生前に試験を失敗させる。
   unsupportedや欠測を有効score 0へ変換しない。通常runtimeの欠測政策とは分ける。

実装時にはfixtureのseed、音源、放射区間、解析支持終端、代表区間、閾値生成則、
失敗順序を取得前に確定して封印する。本案から自動的に閾値や成功条件を選ばない。
既存の選択関数を丸ごと複製せず、実際の選択・拒否・生成経路に限定したtest用入力境界を置く。
通常経路の音声・出生記録が変わらないことと、変更後の全cargo test／fmt／Clippyを確認する。

この一機会が通っても、成人のenergyから親が選ばれる過程、世代を跨ぐ生存差、
非同期配送下の出生保留、有効評価率、作者既定への採用は未検証として残る。
