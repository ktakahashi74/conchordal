# F3b: 自己除去環境の有界worker配送

日付: 2026-09-26。状態: 実装前登録。F3aの解析器を音声生成から切り離し、欠落を隠さず配送する最小単位。
通常runtimeへの接続、身体代表密度のcache、候補生成・移動の変更はこの単位に含めない。
実装はまず隔離worktreeの `cfg(test)` 範囲に置く。

## 固定する範囲

- 48 kHz、hop 512、解析epochと設定はworker生存中に固定する。F3a同様、最大4 source。
- 入力queueは4 frame、出力queueは2 batch。各batchは同一frameの全sourceの結果を含む。
- PCM frame bufferを6組（入力queue 4、処理中1、生成側1）だけ先行確保する。
  一組は混合PCMと4 source分のPCMをそれぞれ512 sample、identityを4個まで保持する。
  submitは既存bufferへのcopyと有界channelの `try_send` だけを使う。sourceごとのVecを毎hop作らない。
- workerは全入力frameを順に処理する。出力queueが満杯なら結果batchを捨て、件数を記録するが、
  入力音響の解析履歴は継続する。結果の欠測とPCM区間の欠落を区別する。
- 入力queue満杯／再利用buffer不足では、そのframeを無音として補わない。
  当該観測epochを無効化し、滞留済みを含む以後の結果を採用しない。自動復活・無言の途中再登録はしない。
- F3aの入力拒否もepoch全体を無効化する。再開には新しい観測起点とworkerが必要となる。共有の失効状態は受信時と判断での採用時に再確認し、
  queue内や受信済みcacheに残る古いbatchも失効後は採用しない。

## 時刻とidentity

frame index・出生sample・source id/generation・epochはF3aと同じ意味を保つ。
提出sampleは呼出し側の明示時計で付け、支持終端より早い提出を拒否する。
workerは処理開始・完了の実 `Instant` を記録する。音声sampleの完了時計を支持終端から捏造しない。
消費側がqueueを実際に受信した呼出し時のsample時計を `received_at_sample` として付ける。
`support_end <= submitted_at <= received_at <= decision_at` を満たさない結果は理由付きで拒否する。
提出時計はframe順に、受信時計と判断時計はそれぞれの呼出し順にepoch内で単調非減少とする。
別batch単体の不等式が成立しても時計逆行は拒否する。constructorでも48 kHz・512 sampleを検査し、
F3a単体が許す8 kHz設定をこの固定版へ無言に受け入れない。

出力を受信できなかった時刻へ遡って適用しない。配送で最も新しい支持終端のbatchを選び、
同じ判断へ異なるframeのsource結果をつぎはぎしない。判断時の支持終端からのageは4,800 sample
（100 ms）以下とする。境界値は許可し、4,801以上をstaleとして拒否する。この値は最初の配送実験の契約であり、
実時間で達成したとの主張でも、最終的な楽器の採用値でもない。

消費側は現在のsource id/generation/birthと結果identityの完全一致を要求する。
退役sourceの結果を後続世代へ流用しない。sourceの自己除去環境は実PCMに依存するため、
身体代表densityのbody generation／recipe／候補identityはこの環境workerへ付け足さない。
それらは次の候補評価段階で別に検査し、両者を同じものと扱わない。

## 検査

1. 実thread/channel経路で1／4 sourceの連続frameを送り、直接Processorと数値・identity・支持終端を比較する。
   threadを動かしただけで、CPU資源や通常runtime配送に合格したとはしない。
2. worker開始を明示的なテストbarrierで止めてinput queueを満たし、次frameでoverflowし全結果が無効になることを検査する。
   schedulerの偶然やsleep時間に失敗検出を依存させない。出力queueを先に満たした状態での
   入力overflowと、受信済みbatchの採用前に起きるF3a入力拒否も検査する。
3. 受信を意図的に止めて出力queueを満たし、dropを計数する。その後の解析結果が直接参照と一致し、
   入力欠落とは違って解析履歴が壊れていないことを確認する。
4. 受信前・未来支持・未来提出・時計逆行・stale境界・世代違い・epoch違い・未対応解析設定を拒否する。
   欠測を有効なscore 0へ置換せず、自己除去環境だけを配送する。
5. 入力bufferの個数・長さ・capacityが定常処理で増えないことを確認する。worker解析器内部のallocationを
   ゼロと主張しない。thread終了は入力切断で行い、callbackからjoinしない。

F3aのsource・数値登録は変更しない。実装後は全cargo testと変更相当のformat/lintを記録する。
I11 §5.7の専有費用測定中はbuild/test/renderを開始せず、静的な実装・レビューだけ進める。
