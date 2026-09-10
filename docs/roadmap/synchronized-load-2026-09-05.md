# 同期した反復発音の負荷試験 — 2026-09-05

[θ時計の位相修正](gate-clock-stability-2026-09-05.md)後、適応時計の`pulse(8)`脚本は
平均約3回/Voice/秒となり、密な反復発音という試験条件を満たさなかった。
同時発音の負荷を制御できるよう、評価ツールに`--timing synchronized`を明示的な選択肢として追加した。

## 実験条件

このアッセイでは同期そのものが必要な実験条件となる。既存の`set_scaffold_shared(16)`でθを固定し、
`pulse(32)`の蓄積と既存の2ゲート間隔制限で8回/秒のonsetを駆動する。
通常の`adaptive`、公開sample、scaffold無効時の生成方式は、この選択肢によって変わらない。
専用の拍打ちVoiceは加えない。固定時計が生成した同期を、音楽的な創発の証拠には使わない。

蓄積率32は発音率ではない。gateの重みが掛かっても反復が続くように駆動し、実際に8回/秒となったかは
reportから別途検査する。最初の蓄積率8の同期試行では、30秒間にVoiceあたり230〜231回程度しか発音せず、
125 msの間隔に約181〜192 msの間隔が混じった。評価ツールは3音色とも拒否した。
この失敗は`target/rt-evaluation/2026-09-05-sync-dense-device/`へ保存した。

`synchronized`は`dense/recovery`と`--reports 1`でだけ使用できる。
各測定窓で以下を検査し、条件不一致の実行を性能合格にはしない。

- 全Voiceが発音し、各Voiceの回数が窓長×8に対して境界の1発分以内に収まる。
- 同じVoiceの同一report hop内に複数onsetがない。
- 同じVoiceのIOIが125 msに対して1 hopと時刻丸め誤差以内に収まる。
- onset recordが`scaffold_mode=shared`を示す。
- 指定した生存Voice数、実機callback、時間進行、割当計測など既存の性能検査も満たす。

onset reportはhopの時刻を記録するため、IOIはsample精度の測定ではない。
成功した回復試験でも、過負荷区間を含む全体の`device_rt_pass`は`null`とし、
baseline・過負荷・回復の窓を個別に評価する。

## 検査で見つかったframe境界の欠落

θ位相がちょうど0のframe先頭では、その境界が前のframeの範囲外かつ次のframeの「未来」でもないため、
実際の周期境界が取り落とされていた。`ThetaGateClock`がframe先頭の境界を含むよう修正した。
同じframeを再処理した場合には、前回以下の境界を再発行しない。
低いsample rateの回帰で取り落としを先に再現し、frame先頭とframe内の境界が残り、
再処理では古い境界が出ないことを確認した。

これは先行する位相・音長・終了音追跡修正の後の変更である。
先行する10分の成功記録は、その時点のバイナリに結び付けて保持する。

## 64 Voice・3音色

保存先は`target/rt-evaluation/2026-09-05-sync-dense-verified/`。
seed 1、report有効、DCC 0.25、warmup 5秒、測定30秒、MOTU M2、48 kHz・hop 512・ring 4800 frames。
設定や機器の優先度を変更せず、3条件を順番に実行した。

| 音色 | hop p99 ms | 最大hop ms | 出力不足 frames | 発音回数 / Voice / 秒 | 判定 |
|---|---:|---:|---:|---:|---|
| sine | 1.281 | 1.578 | 0 | 8.000 | 成立 |
| harmonic | 1.321 | 2.250 | 0 | 8.000 | 成立 |
| modal | 1.449 | 1.942 | 0 | 8.000 | 成立 |

3条件とも全64 Voiceの回数とIOI検査を満たした。
この同期負荷条件での短時間の性能確認であり、任意の音色設定やVoice数、長時間演奏へは一般化しない。
バイナリのSHA-256は`a8802f17bd956e4d2203dfc78d2b607daa96e353f5ced112acb34ea857fee525`。

## 負荷超過と回復

同じバイナリと設定でharmonicの16 Voiceを10秒測定し、集団を5秒間増やした後、
追加分をreleaseした。2秒の待機を挟んで16 Voiceを10秒測定した。
最大Voice数に合わせた振幅を全区間で維持し、回復時だけ音量を変更する操作は加えていない。

| 試行 | 区間 | Voice数 | hop p99 ms | 最大hop ms | 出力不足 frames |
|---|---|---:|---:|---:|---:|
| 512 | baseline | 16 | 0.352 | 0.932 | 0 |
| 512 | 負荷 | 512 | 7.854 | 11.602 | 0 |
| 512 | 回復 | 16 | 0.374 | 0.461 | 0 |
| 1024 | baseline | 16 | 0.570 | 0.960 | 0 |
| 1024 | 負荷 | 1024 | 20.561 | 24.539 | 107,520 |
| 1024 | 回復 | 16 | 0.823 | 1.246 | 0 |

512 Voiceの試行では、p99がhop予算10.667 msを超えず、出力不足もなかった。
最大hopの単発超過だけでは今回の過負荷条件に該当しないため、`overload_observed=false`、
`recovery_pass=null`とした。1024 Voiceでは両方の超過を観測し、回復窓が再び性能条件を
満たしたため、`overload_observed=true`、`recovery_pass=true`となった。
出力不足はcallbackが要求した音声をringbufferから取得できなかったframesであり、機器のxrun数ではない。

全測定窓でVoice数、発音回数、IOI検査を満たした。baselineと回復は8回/Voice/秒。
負荷窓は新規Voiceの最初の発音が窓内へずれるため、平均7.806回と7.803回となった。
新規Voiceあたり39回、既存Voiceあたり40回で、境界の1発分という許容内に収まる。

保存先は`target/rt-evaluation/2026-09-05-sync-recovery-512/`と
`target/rt-evaluation/2026-09-05-sync-recovery-1024/`。
この5秒間の負荷試験から、512 Voiceの長時間対応や任意の過負荷からの回復を保証しない。

## 証拠と検証範囲

実行中のPIDとprofileパスを照合し、PipeWireでMOTU M2へのactive linkを別途観測した。
成功した3音色の試行は計22回、512 Voiceは6回、1024 Voiceは2回の観測を保存した。
1024 Voiceの観測開始は実行後半となったため、過負荷区間を通して接続先を確認した証拠にはならない。
他の試行についても接続観測は連続監視ではない。runner自体の機器名だけによる未検証表示は保持する。

失敗した同期試行を含む4 campaignのsource 177ファイル、設定、バイナリ、case成果物のhashを検査し、
profileとonset reportから判定を再計算した。現行の測定対象sourceも最後のsnapshotと一致した。
楽器による音声ファイルの生成はない。
検証記録は`target/rt-evaluation/2026-09-05-synchronized-load-validation/validation.json`。
同じディレクトリにframe境界の修正前の失敗、Rustテスト全出力と終了値、Python・Clippyの記録を保存した。
通常Rustテスト664件、Python比較ツール46件、Clippy・format・差分検査が通過した。

後続の[harmonic・64 Voiceの600秒試験](synchronized-long-harmonic-2026-09-05.md)でも、
同じバイナリで8回/秒の回数・間隔条件を満たした。測定窓とrelease中の出力不足はゼロ、
hop p99は1.592 msだった。先行する約3回/秒の記録とは別の証拠として保持する。
次は小節アクセントの生成へ進む。
