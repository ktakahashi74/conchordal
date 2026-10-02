# 非Sine身体・control変更の通常runtime実時間結果（2026-09-27）

[取得前登録](body-fitness-non-sine-changes-runtime-live-registration-20260927.md)の専有取得は、OFF合格・ON不合格だった。合否閾値を変えず、初回のON不合格を保持する。全体suiteの合格とは別の判定である。

## 固定条件と計測結果

第十三版bの通常runtimeで、Harmonic/Modal各1声、seed 17、48 kHz/512、10秒・938 hopをON/OFF各一回取得した。Harmonicのbrightness変更はframe 300、control変更はframe 600。代謝flagは両方OFF。実audio deviceを開かず、512 sampleの模擬ringbuffer消費先へ送ったPCMを破棄した。

| action | p50 hop ms | p95 | p99 | 最大 | underflow | 10.6667 ms超過 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OFF | 0.285897 | 0.573236 | 0.661641 | 1.091196 | 0 | 0 | 合格 |
| ON | 0.270701 | 0.668138 | 0.907013 | 1.183347 | 0 | 0 | 不合格: 身体変更後の回復期限 |

両条件とも938 hop全体で2声を維持し、targetと実基音が両声で変化した。ONのobserverは最後の定期報告frame 912までに912 batchを受理し、停止なし。消費時の支持ageとscore使用、各sourceのcache上限256 entry・1 MiBを満たした。終点近傍のcacheは各727,040 byte。この値はprocess全体のメモリ量ではない。worker hop費用は背景の密度準備workerのCPU費用を含まない。

## 不合格の内容

| 定期報告frame | Harmonic Recipe世代 | Harmonic累積消費 | Modal累積消費 |
| ---: | ---: | ---: | ---: |
| 288 | 1 | 3 | 5 |
| 432 | 2 | 3 | 13 |
| 576 | 2 | 4 | 19 |
| 720 | 2 | 20 | 24 |
| 912 | 2 | 46 | 40 |

登録した身体回復判定はframe 432までの新世代評価消費を要求した。世代は2へ変わったが、消費数は3のままで、最後に消費した値も旧世代だった。初めて新世代を消費した記録はframe 528の報告に現れ、receiptのdecision sampleは267264、つまり実消費frame 522。変更から222 hop、2.368秒であり、登録期限frame 432を90 hop超えた。後で回復した事実を期限内合格に読み替えない。

control変更後はframe 720までに消費が増え、frame 912まで継続した。`Consume(ControlChanged)` の拒否記録も1件ある。ModalのRecipe世代は1のまま。ONの実target変更はHarmonic 47回・Modal 41回、実基音変更は466回・579回。これらは主窓全体のprobe計数であり、frame 912の累積消費46/40とは集計時刻が異なる。

読み取り監査では、frame 336→384でglobal completedが13→14、Harmonicのcacheが196→254 entry、invalidations 0、Modalのcacheは239 entryのまま、Harmonic消費は3のままだった。旧世代のHarmonic jobがこの区間で完了し、世代不一致で破棄されたと判断できる。その後の新世代jobはframe 480まで未完了、frame 528ではcache 132 entry・invalidations 1となった。新世代の冷計算は132候補×72 hop、9,504 Tone/主観密度解析frame相当である。

現在のaction配線は身体変更でreadyを破棄するが、pending jobを中断せず、その終了まで次のrequestを投入しない。cacheは世代・brightness変更で全消去する。brightness 0.30の密度を0.85へそのまま流用する修正はidentity契約に反する。Harmonic固有の拒否はこの身体回復区間で0。後のcontrol拒否は遅延原因ではない。request/receiveの正確な時刻は現reportに無いため、旧job待ちと新job所要時間の厳密な分解はできない。

次の改善候補は、旧世代計算の協調的取消、submit/完了/破棄時刻の診断、冷計算の負荷削減である。取消だけで今回の期限に達すると断定しない。別版・別登録で候補scoreの完全性、鮮度、取消時のcacheとRNG不変を維持して検査する。

## 固定した証拠と範囲

生記録、登録の写し、実行command、ホストの実行前後の競合process 0確認、全出力、exit 101は `target/runtime-non-sine-v13b-20260927/` に保存した。`raw/manifest.json` が各scenario/IR/config/report/profileと集計を参照する。保存したrelease test binaryのSHA-256は `5e147c231ba7d272ffeb139e548a44aa6afcbd68e78148acc75a2f0270ea140f`。405 sourceファイルのmanifestは `target/integration-source-20260927-v13b/manifest.json`、SHA-256は `2bc58ef9bffdb12eb50c042a1bcc678d9197d2d081f9d27d60b67bd263823b61`。

実時間の音切れ・hop費用はこの一取得の範囲で条件を満たしたが、非Sine身体変更後の追随は登録未達。実device、可聴品質、長時間や長期生態の受入ではない。改善と再取得は別版・別取得として扱う。
