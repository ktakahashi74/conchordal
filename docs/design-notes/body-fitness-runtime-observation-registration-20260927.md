# 身体評価の通常runtime観測接続

2026-09-27。取得前登録。統合第八版を起点に `.worktrees/body-fitness-runtime` で実装する。F3eの提案延期とF4dの出生検査は別worktreeで進め、ここでは音声から自己除去環境までの通常ビルド経路を接続する。身体評価による行動変更はまだ行わない。

TOMLのルートに実験用 `body_fitness_observation = true` を指定した場合だけ有効にする。省略時はfalseで、従来の動作を保持する。初版は48 kHz・512 sample/hop・同時4 source以下。設定や実機sample rateが範囲外なら開始前に拒否する。SourceIdentityはid・generation・出生sampleを持ち、自己PCMの捕捉は出生hopのrender前から行う。欠測をゼロで埋めず、入力のgap・容量超過・worker飽和・解析設定更新は理由付きで観測を無効化する。

共有の通常workerが実ScheduleRendererから得た混合PCMとsource別habitat PCMを、既存の有界source-removed workerへ毎hop送る。instrumentはnonblocking受信だけを使う。offline renderは明示的に決定的配送を選び、worker完了を待つ。この二つの配送modeを診断に明記し、offline取得を壁時計性能と呼ばない。支持終端・受信・判断・epoch・sourceの既存accept境界は維持する。生成側へ新しいscoreを渡さない。

`--report`へ間引いた `body_fitness_observation` recordを出し、受理数、source別の自己除去mass、支持時刻、配送mode、無効化理由を保存する。既存のUIや作者操作は追加しない。

通常のconchordal-render binaryを使い、seed 7、48 kHz、1/4 sourceの固定身体・持続発声をON/OFFで比較する。WAV byteと既存のonset/population/death/spawn/respawn記録を一致させる。1 sourceは自己除去後mass 0、4 sourceは他者のmassが正、受理数が正、support/receive/decisionが合法、最大age 4800 sample以内を要求する。5 sourceを容量超過として明示拒否する負例と、48 kHz/512以外の設定拒否も検査する。通常build経路を通した検証であり、cfg(test)のshadow hookを証拠に使わない。

全cargo testとログ・同一shellの終了値、fmt、標準Clippy、全target checkを記録する。身体評価の提案消費、代謝・出生、非同期密度準備、有効評価率、音声deviceの資源受入、main既定への採用はこの単位の完了に含めない。
