# M0: accentの有界保存と一度だけの配送

2026-09-12。Rustのaccent ledgerを実装し、既存の四hop detectorからの遅延配送を接続検査した。
`cfg(test)`の数値部品であり、本番workerやFrontendの全group lifecycleへはまだ組み込んでいない。
[source・参照・検査の記録](m0-accent-ledger.json)を保存する。M0全体は未完了である。

## 原支持と配送

accentは元のgroup handle、不確実なevent区間、四つのcanonical raw区間、
全source開始・終端、利用可能時刻、正のweightを保持する。
受領時刻が全証拠の利用可能時刻に達する前は配送しない。
配送時には別の受領時刻とgeneration内の単調なsequenceを付け、元時刻を書き換えない。
generationやbusの異なる入力はcreditを増やさない。

同一IDの直近または保存中の再送は、内容が一致すれば何もしない。
内容が違えばエラーとし、元のbank・時計・累積値を変更しない。
保存期間外の古いIDを右側支持の延長によって再採用できないよう、
原event終端とstable IDの組、および全raw証拠の順序keyを、いずれも厳密な単調順序に制限する。
任意のpacket並べ替えやgroup間の祖先解決は、このledgerの責務に含めない。

## 容量と過去snapshot

既定の登録条件は一groupあたり128 accents・32秒である。Rust部品には容量と物理windowのsample数を
明示し、64／128／256の容量感度条件を検査する。構築時にbankを確保し、配送・時間更新では増やさない。
物理windowより古いeventを先に消し、まだbankが満杯なら最古のevent時刻・stable IDを除く。
weightの大きさで残すaccentを選ばない。

容量によって失った最大event時刻を保持し、その時刻を含む密度windowのweightをunknownとする。
時間経過による通常の期限切れは、この容量欠落watermarkを置き換えない。
累積配送数と累積weightはbank保持から独立する。遅く受領したため既にwindow外のaccentも一度だけ数える。
この累積値は取得coverageや初回accent支持を代替しない。空bankだけから「観測された無変化」とは判定できない。

明示的なsnapshot要求はaccentの独立したVecコピーを作る。後続配送・期限切れ・evictionによって、
既に返したsnapshotを書き換えない。snapshotコピーとallocationの費用は、allocationを行わない配送経路と
分けて測る。全beamが保持するsnapshot数や全O04の配置を検証したものではない。

## 参照の反例と修正

旧Python参照では、同じstable IDの右側raw支持を延ばして再送すると、二度目のcreditが入った。
元の参照、初回・二回目の配送、誤った累積値を保存した。
ID一致の検査を証拠順序の比較より先に行い、保存外の同時刻IDもevent／ID順序で拒否するよう修正した。
仕様9.2の「順序付き配送」を明文化し、20登録の現仕様SHAを更新した。
過去のsource・検査・archiveのhashは保持し、accent登録の現sourceと規約だけを追加更新した。

既存のassigned energy、accent detector、causal cut、AccentStreamはsource単位で不変である。
前段raw-feature fixtureの全1,152入力・期待出力も不変で、参照全文SHAだけを現版へ更新した。
既存のfit layoutや数値係数は変更していない。

## 検査と残務

Python参照の四容量条件・1,308操作について、配送結果、全bank、複数密度window、容量mask、累積値を照合する。
Rust固有の検査は、遅延配送、凍結snapshot、同時刻ID、延長された原支持、容量欠落と通常expiryの区別、
不正入力、sequence上限、generation変更を含む。
実際の四hop数値detectorから20 accentsを渡し、元時刻を保持した一度だけの配送と、全expiry後の累積値を検査する。
この入力は合成energy／scanであり、PCMからの独立したaccent回収検査ではない。

全Rust検査は858成功・0失敗・23 ignored、今回の追加は6検査である。
Pythonのaccent・descriptor・section関連96検査も成功した。
1,308操作の内訳は配送1,204、状態不変または時計更新72、期待された拒否32である。
保持windowの開始境界ではeventを残し、その1 sample後に期限切れとなることも確認した。
fixture読取りの最初の失敗はPythonの初期時刻`0.0`とRustの整数sample表現の違いであり、
generatorの時刻表現を整数へ揃えた。数値規約・期待結果は変えず、初回の失敗ログも保存した。

releaseで一つの飽和bankを各6,000回更新した部分測定は次のとおりである。

| 容量 | 配送p99 | snapshotコピーp99 | bank payload |
|---|---:|---:|---:|
| 64 | 0.13 µs | 1.06 µs | 8,704 bytes |
| 128 | 0.09 µs | 1.16 µs | 17,408 bytes |
| 256 | 0.14 µs | 2.24 µs | 34,816 bytes |

ledger headerは256 bytes、snapshot headerは120 bytesであり、全bankを写したsnapshotのpayloadは
表のbank payloadと等しかった。snapshot計測はコピーとallocationを含み、破棄を含まない。
原音処理、period pair、beam、全worker、実音声出力、全RSS／stackを含めた測定ではない。

Rustへ移したbankはまだperiod pair-gridやarrival clockへ接続していない。
pairの正確な除去、観測coverage、beam／contextの共有、退役groupの保存と祖先処理、
全負荷・全fit・人のgateは残る。これらをbank単体の成功で完了扱いしない。
