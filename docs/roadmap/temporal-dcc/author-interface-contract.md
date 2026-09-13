# W0: 時間認知の操作・表示・試聴契約

日付: 2026-09-12。状態: 使用仕様。`temporal_mode("off"|"observe")`は初期観測まで実装した。
dispositionと`participate`は未実装。実行できる例と検証範囲は[m1-observation.md](m1-observation.md)を参照する。
正本は[完成計画](../temporal-dcc-completion.md)と[採用した工程改訂の§5.1](implementation-plan-revision.md#51-scenariorhai-apiとuiuxの最小範囲)。
識別子の最終固定はM8で行う。

2026-09-13の[採用スコープ](milestones.md#11-採用スコープと認知機構の選定2026-09-13)に従い、
I9/T7の長期聴取・全曲完了感は研究拡張とする。Scenarioは構成と環境条件を与え、
ListenerTwinには実際のpresentationの証拠だけを渡す。全曲記憶の有効化や全曲完了感を通常操作の前提にしない。
終端を持たない継続運転は設計対象であり、今回の文書変更で対応APIや通常運用の検証を追加したとはしない。

## 設定の所有者と解決

| 設定案 | 所有者・対象 | 単位・範囲・既定 | 宣言と変更 |
|---|---|---|---|
| `temporal_mode("observe")` | Scenarioの開始前設定。演奏全体 | `off`、`observe`、`participate`。省略は`off` | top-levelで時刻0の宣言を一度だけ許す。同値を含む重複はエラー。section／時間付きclosure内や時刻進行後は開始前設定へ読み替えず拒否 |
| `.temporal_disposition(#{ continuation: 0.2 })` | PopulationSpec。そこから生成するPopulation／Voiceの初期傾向 | §9.6に列挙した帰結への無次元・有限の符号付きgain。各項目の既定0。非有限値と未知keyはエラー。追加のclip範囲は設けない | 生成前のbuilderで指定。同一builderでの重複指定はエラー。`variant()`は明示した初期値を複製 |
| `population.temporal_disposition(...)` | 既存Populationへの時刻付き更新の案 | 同じ単位・key。指定keyだけを更新 | M6の実消費者の成立後に追加。指定時刻以後の候補だけに適用し、発行済み予測・既存観測を変更しない |

初版では追加設定をScenarioだけに置き、新しいTOML設定は設けない。既存configは身体・解析等を
引き続き所有する。Rhaiとconfigの二重設定・優先順位は発生させず、GUIも設定を書き換えない。
既存TOMLの未知key拒否を維持する。Populationへmode設定は追加しない。

M1では`off`と観測可能な範囲を実装し、`participate`は対応する生成経路が実装されるまで起動前に
拒否する。未実装keyも拒否する。実装済みだが未校正・支持不足の帰結は設定を保持し、該当する
作用だけを無効にして理由を返す。一つの未校正headによって全ての身体動作を停止しない。
listener由来gainは別の設定として扱い、初期値0を維持する。

## 最小構文案

これは未実装の構文を含む設計例であり、`samples/`へ実行可能な例として登録しない。

```rhai
temporal_mode("observe");
let voice = sine()
    .amp(0.08)
    .sustain()
    .temporal_disposition(#{ continuation: 0.2 });

section("a single voice", || {
    let v = place(voice, at(220.0));
    wait(4.0);
    release(v);
    wait(2.0);
});
```

`observe`ではgain指定があっても新しい生成作用はゼロ。`section()`は作者の構成ラベルであり、
Listenerが推定するsectionへの入力にはならない。継続へのgainと、作者が指定した持続・音長は
異なる操作である。modeを変える場合は現在の演奏を終え、新しいepochと空の記憶から開始する。

## 共通診断と表示

| 表示 | 必要な状態・理由 | 所有者 |
|---|---|---|
| 設定 | 既存動作／観測のみ／生成に反映。解決したmodeとPopulation別設定 | Scenario／Populationの解決済み設定 |
| 観測 | 未開始、待機、受信、欠落、遅延。bus、epoch、元支持、利用可能時刻、受信時刻 | 配送と認知の共通診断 |
| 関係 | 未確定、候補、支持あり。未実装・未知と既知ゼロを区別 | 認知モデルの契約adapter |
| 作用 | 設定による無効、未実装、未校正、支持不足、期限切れ、適用可能、実際に選択・実行 | 実際の消費者と行動記録 |

UIとreportは同じ計算済み要約を読む。画面専用の閾値・認知判定は作らない。GUIなしでも同じ理由を
得られ、reportや画面の有無・更新間引きで認知と生成が変わらない。古いsnapshotを現在の支持として
表示しない。全beam・全bankのframeコピーは行わず、診断のコピー費用もO04/O17で計測する。
M1のfixtureには、各状態、旧epoch、欠落後の再開、EOF、順序逆転、受領前の未来支持、report／GUIの
有無を含める。MR3は最初の実際のモデル・特徴・fit・保存形式の利用先から追加する。

## 試聴と受入

比較する問いを一つ決め、条件・source・model・seed・素材と音量の扱いを保存する。音源は
`conchordal-render`で作り、必要な先行文脈も聞く。モデル名を伏せたA/Bと、同じ／判別不能の回答を
許す。同じseedだけの別runは探索的比較とし、O15の入力・身体・候補を揃えた状態介入と区別する。
診断表示を見る順序も比較前に決める。作者の制作上の試聴を独立した認知課題の代わりにはしない。

M1は表示と内部状態の一致・生成作用ゼロ、M6は実消費者と時刻付き更新、M8–M9は日英文書・
生成API参照・実行sample・既定off・資源・音の回帰を検査する。専用比較アプリやライブmode切替は
初版の前提にしない。初期観測の自動検査を超える操作・表示・試聴の受入は未実施である。
