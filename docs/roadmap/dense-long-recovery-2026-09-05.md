# 反復発音・長時間演奏・過負荷後の回復 — 2026-09-05

後続の[θ時計と終了音追跡の調査](gate-clock-stability-2026-09-05.md)で、以下の試行には
位相更新漏れによる異常な発音集中が含まれると判明した。数値と当時の判定は履歴として保持するが、
正常な高密度発音やその過負荷・回復の対応範囲を示す証拠としては採用しない。
`pulse(8)`は固定8 Hzではなく、適応θ時計上の蓄積率指定だった。以下の「反復発音」はこの脚本を指す。

既存の固定sustain集団の36条件に続き、短い音の反復と一時的な負荷増大を実機で測定した。
短時間の64 Voice反復試験では、sine・harmonicが性能条件を満たし、modalには出力不足があった。
16→512→16 Voiceの試験では負荷超過を観測し、負荷を戻した後の測定窓で回復を確認した。
初回の10分試験ではScenario時計の累積誤差も検出し、実装を修正した。
時計修正後の10分試験では全測定窓で64 Voiceを維持したが、出力不足4,096 framesで性能条件は未達だった。
modal再試験では16 Voiceが条件を満たし、64 Voiceでは不足が再発した。

## 比較手順の追加

`scripts/evaluate_rt.py`へ`--workload dense/recovery`とreport・DCC条件の絞り込みを追加した。
既定のsustain・36条件の比較は維持する。実行例と判定の契約は
[負荷比較手順](runtime-load-evaluation.md)を参照する。

- dense/recoveryは`pulse(8)`の蓄積率、1周期の保持、release 50 msを指定する。
- 脚本のHz指定だけで成功とせず、各窓で全Voiceが発音し、平均4回/Voice/秒以上あることをreportで検査する。
- recoveryはbaseline・負荷区間・回復窓を分け、生存Voice数を各窓の全hopで検査する。
- 負荷超過なしは回復未判定。回復窓の出力不足やp99超過を成功とは扱わない。
- 生成した脚本はseedを固定した性能assayであり、作品sampleに専用の拍打ちを追加していない。
- 楽器`conchordal`を使い、音声ファイルは保存していない。

Python比較ツールの検査42件、時計修正後の通常Rust検査660件が成功した。
最終の小変更後、RT用21件も再実行して成功した。Rustの全出力・終了値は
`test_report.txt`・`test_status.txt`へ保存した。Clippyも通過した。

## 共通条件

AMD Ryzen 9 9950X、MOTU M2、48 kHz、2 channels、hop 512（10.667 ms）、ring 4800 mono frames。
releaseと`profile-alloc`を使用した。seed 1、report有効、DCC 0.25、warmup 5秒。
DCC 0.25はこの負荷比較の条件であり、推奨値への変更ではない。
各campaignにcommit、未コミット差分、source、設定、バイナリ、各成果物のハッシュを保存した。
基準commitは`a5447e1444c1224106fdeb5115bfca8ed84e4470`とローカル差分である。

自分のビルド・テスト・別の実機試験を同時に走らせず、条件を逐次実行した。
デスクトップ環境の他のプロセスを停止したり、優先度・CPU governor・機器設定を変更したりしていない。
回復試験では5秒間隔の観測でconchordalから`hw:M2`へのactive linkを確認した。
この経路観測は連続監視ではなく、初回のdense 3条件については経路観測ファイルが得られていない。

## 64 Voice・20秒の反復発音

保存先: `target/rt-evaluation/2026-09-05-dense-device/`

```bash
python3 scripts/evaluate_rt.py --workload dense --voices 64 \
  --bodies sine harmonic modal --reports 1 --dcc 0.25 --duration-sec 20 \
  --output target/rt-evaluation/2026-09-05-dense-device
```

| 音色 | hop p99 ms | 最大hop ms | 出力不足 frames | 平均発音数 / Voice / 秒 | 性能判定 |
|---|---:|---:|---:|---:|---|
| sine | 5.899 | 18.321 | 0 | 7.5 | 今回の窓で条件を満たした |
| harmonic | 8.295 | 12.399 | 0 | 7.7 | 今回の窓で条件を満たした |
| modal | 5.495 | 29.089 | 512 | 6.7 | 未達 |

modalの平均発音数は6.7回/Voice/秒。3音色とも全64 Voiceの反復発音を確認した。
modalでは生成時刻19.018667〜19.061333秒付近で約19〜29 msのhopが連続し、
19.061333秒の累積差として512 framesの不足が記録された。
その際のanalysis待ちは約0.77〜1.03 ms、listener待ちは約0〜0.013 msで、
計測された時間の大半はそれ以外にあった。これだけでOSの割込みや特定のDSP関数を原因と断定しない。
単一hopの最大超過はsineにもあったが、ringの余裕で吸収できる場合がある。
p99・最大値・出力不足を別々に評価する。

## 負荷増大と回復

baselineは16 Voiceで10秒、負荷区間は5秒。追加集団をreleaseして2秒待ち、16 Voiceで10秒測定した。
振幅は負荷区間の最大Voice数に合わせ、前後でも同じ値を維持した。

| 負荷区間のVoice数 | 区間 | hop p99 ms | 出力不足 frames | 平均発音数 / Voice / 秒 |
|---:|---|---:|---:|---:|
| 256 | baseline | 0.815 | 0 | 7.444 |
| 256 | 負荷区間 | 9.881 | 0 | 5.810 |
| 256 | 回復窓 | 0.885 | 0 | 7.700 |
| 512 | baseline | 0.733 | 0 | 7.363 |
| 512 | 負荷区間 | 32.380 | 5632 | 7.098 |
| 512 | 回復窓 | 0.892 | 0 | 7.525 |

256 Voiceでは負荷超過が起きず、`recovery_pass=null`とした。512 Voiceでは負荷超過が成立し、
baselineと回復窓が性能条件を満たしたので`recovery_pass=true`となった。
これはrelease後2秒待った測定窓の回復であり、無瞬断の動作や負荷区間全体の合格ではない。
不足framesはcallbackのring不足によるゼロ補完数で、hardware xrunの回数や長さではない。

保存先:

- `target/rt-evaluation/2026-09-05-recovery-device/`（256 Voice）
- `target/rt-evaluation/2026-09-05-recovery-512-device/`（512 Voice）

## 長時間と追加確認

64 Voice・harmonic、600秒の反復発音を
`target/rt-evaluation/2026-09-05-dense-long-device/`で実行した。
releaseは605秒予定だったが、すでに604.5867秒には全Voiceが消えており、
要求窓に完全に含まれる末尾38 hopで生存数が0になった。
計測ツールは条件不一致として拒否した。この実行を64 Voice・600秒の有効な測定値には採用しない。
エネルギー枯渇ではなく、下記のScenario時刻の累積誤差による早期releaseだった。

`WorkerState`の`current_time: f32`を毎hop加算する経路を除去し、Scenario dispatch・UI・ログが
音声フレームとTimebaseから求めた時刻を使うようにした。
605秒の操作が直前のframeでは発火せず、605秒に達した最初のframeで発火する回帰テストを追加した。
テストは実際の`advance_population`とConductorを通す。
時計修正前の上記短時間・回復測定は元のsnapshotと結び付けて保持し、修正後の値で上書きしない。

修正後の音声バイナリのSHA-256は`4a6334c82b40962b`で始まる。修正前は`94607a1a8124bf9f`。
修正版の10分試験では、測定窓5.002667〜604.992秒の56,249 hopすべてで64 Voiceを維持した。
releaseの予定に達する最初のhopは605.0027秒となり、release尾部後の605.04535秒に生存数が0になった。

| 修正版の条件 | 測定秒 | hop p99 ms | 最大hop ms | 出力不足 frames | 判定 |
|---|---:|---:|---:|---:|---|
| harmonic 64 Voice | 600 | 6.016 | 32.863 | 4096 | 未達 |
| modal 16 Voice | 30 | 1.046 | 1.874 | 0 | 今回の窓で条件を満たした |
| modal 64 Voice | 30 | 7.110 | 31.428 | 1024 | 未達 |

harmonicの実発音は平均7.276回/Voice/秒。1分ごとのp99は3.115〜7.146 msに収まるが、
258.805秒、473.824秒、502.400秒、528.480秒、562.560秒付近で不足を記録した。
全体のp99が予算内でも、連続する遅いhopをringが吸収しきれないことがある。
modalは修正前後の両試行で19.061333秒に不足を記録し、修正版では26.304秒にも不足した。
時計修正だけで出力不足が解消したとは判断しない。

長時間試験のrelease時には測定窓外で約120.526 msのhopと追加1,536 framesの不足があった。
最終のdevice累積値には、演奏終了後のreport集計中も動くcallbackの不足が含まれる。
これを演奏中の欠落へ加算せず、hop内の累積差、release時の不足、終了後の集計期間を区別する。

修正版の10分試験では122回の経路観測がすべてactiveなM2接続を示し、観測エラーはなかった。
RSSの観測範囲は43.8〜90.2 MiB。workerの割当要求は平均約107回・49.7 kB/hopだった。
RSSにはprofile行とreport用の保持データが含まれ、割当bytesは保持量ではない。
今回の増加だけでリークと判断せず、reportなしの長時間試験や他threadの割当評価は残す。

修正版でも16→512→16 Voiceの回復を再確認した。
baselineのp99は0.767 ms・不足ゼロ、負荷区間は32.736 ms・5,120 framesの不足、
release後2秒待った回復窓は0.836 ms・不足ゼロ。callbackエラーはゼロで、回復条件を満たした。

修正版の保存先:

- `target/rt-evaluation/2026-09-05-dense-long-clock-fixed/`
- `target/rt-evaluation/2026-09-05-modal-clock-fixed/`
- `target/rt-evaluation/2026-09-05-recovery-512-clock-fixed/`

[統合検証記録](../../target/rt-evaluation/2026-09-05-stress-validation/validation.json)に、
7 campaignのハッシュ照合、1分ごとの測定、経路・RSS、テスト結果を保存した。
9条件の有効な測定と、時計不具合を検出した1条件の拒否を区別して記録している。
全campaignでsource 177ファイル、設定、バイナリ、各成果物のハッシュが一致し、WAVは作成されていない。

## 次の改善

今回の検証と時計修正は完了した。密な発音で64 Voiceを安定動作の範囲として扱う条件は未達のままとする。
modal 16 Voiceの30秒の成功を、全音色・10分以上の範囲へ一般化しない。
次は19秒付近のmodalの遅延と、長時間演奏のrelease時の遅延を対象に、
生態更新・合成・report処理の時間を分け、原因を特定してから修正する。
小節アクセントの拡張より先に、この出力不足の安定化を扱う。
