# 演奏の場 conchordal.org/demo/ の設計（コンセプト・要件・設計）

作成：2026-11-16（統括、Claude Opus）。同日、作者の言葉「.space は、将来、作品を共有する SNS にする予定だった」を受けて、置き場所を conchordal.space から conchordal.org に改めた。状態：S-1〜S-3b を branch `web/demo-20261116` に実装した（未 commit）。配備と公開はまだ行っていない。実装の段で決めたことは §12 にまとめた。

前提の記録は次のとおり。
- 台帳[§9.3.129](technote-ledger.ja.md#art-form-path-20261106)（ライブのみ、録音を公開しない、台本で配る、browser の版を既定の配布の候補にする）。
- 台帳[§9.3.130](technote-ledger.ja.md#browser-numerics-20261106)（native との bit 一致を求めない）。
- 台帳[§9.3.132](technote-ledger.ja.md#first-impression-cue-response-20261107)（第一印象に足る曲の要求）。
- browser の版の作りは `crates/conchordal-web/README.md`。計画と測定の記録は `.orchestration/units/` の WP-0〜WP-6、WA-1、OP-1〜OP-3。

## 0. 要約

- 次版の公開に合わせて、`conchordal.org/demo/`（英語だけ。2026-11-20 に作者が日本語の page を外した）に、Conchordal の作品を聴き手の browser の中で生で演奏する場を置く。
- conchordal.space は、将来の作品を共有する SNS のために空けておく。この場の「一つの作品を生で演奏する単位」は、SNS が作品の page として再利用できる形で作る（§11）。
- 録音は置かない。音は手元で生まれ、どこにも保存・送信されず、演奏が終われば消える。配るのは台本と楽器（wasm）だけである。
- 台本は折りたたまずに、楽器のすぐ下に見せる。短い台本と生態から音楽が生まれることを、その場で見せるためである。
- 性能の改善のため、演奏の性能の数値（音切れの数、一 hop の計算時間など）を送る。送ることと送る中身を Start の近くに表示し、止められるようにする。音と、音から作った値は送らない。
- engine・GUI・台本は release の tag から CI で build して Release の asset にし、org の配備はその asset を取り込む。文書を直すたびに wasm を build し直すことはなく、演奏の場は版に固定される。
- 基盤は WP-5／WP-5R の試作（org の Astro の中の `/demo/`）である。
- 載せる曲は作者が採った作品だけにする。sample 01〜16 は載せない（§9.3.132 で第一印象に足りないと判定済み）。
- いま作者に要る判断は 0 件。公開の前に要る判断と作業は §8 にまとめた。

## 1. 前提（既に決まっていること）

- **§9.3.129。** デモはライブで、録音は公開しない。作品は台本として配る。browser の版は既定の配布の候補で、採るのは対象機、実際の音声経路、停止と再開、native との差を確かめてからとする。Air-Gap については、browser の版でも音を保存・送出する経路が無いことを実際に確かめる。共有世界は別の候補である。
- **§9.3.130。** browser の版は native と bit 一致しなくてよい。受入は、同じ source で動くこと、出力が有限であること、音切れが 0 であることで判定する。
- **§9.3.132 と、その後の採択。** 第一印象に足る曲には四つの要求がある（両方の DCC など）。作者は 11-14 に一つの版を「現時点の一つのデモとしていい」と採った。その後の版（形 P、TD-4、MO 系）は試聴と判断の途中である。
- **既にあるもの。**
  - engine：`StepEngine`（feature `web`）、Web Worker、AudioWorklet、egui の GUI（WebGL）。main に統合済み（`317e7931`）。
  - 速さ：OP-2・OP-3（main の `72023c8e`、`04c6ba7e`）の後、TD-4 の台本の一 hop は、blanche の headless Chrome で平均 2.25 ms、p99 3.5 ms だった（予算は 10.667 ms）。改善前の M5 MacBook Air では平均 6.33 ms、p99 18.87 ms で、約 10 秒続いた機械の側の遅れ（約 1.7 倍）の間に音切れが出た（WA-1、OP-1）。
  - ページの試作：WP-5／WP-5R が org の Astro に `/demo/` と `/ja/demo/` を作った。worktree は `.worktrees/wp5-demo-20261106`（基準は `317e7931`、未 commit、非公開）で、内容は「Here and now／いま、ここで」の構成、曲の切り替えの直列化、全画面、GUI の縮尺、到達点の表である。
- **domain（2026-11-16 に確認）。**
  - conchordal.org：nameserver は Cloudflare。site は GitHub Pages で、Cloudflare の proxy を通して配る（`web/static/CNAME` は `www.conchordal.org`）。HTML は br で圧縮され、`Cache-Control: max-age=600` が付く。
  - conchordal.space：Cloudflare に登録済み（2025-11-06 登録、期限 2027-11-06）。DNS は Cloudflare の proxy を指すが、HTTPS は応答しない（origin が無い）。将来の SNS のために使わずに置く。

## 2. コンセプト

**一文。** conchordal.org/demo/ は、Conchordal の作品が聴き手の手元で一回ずつ演奏される場である。開くたびに新しい演奏が始まり、録音されず、終われば消える。

この一文は設計の上での言い回しである。ページの文言は作者名義の文章として実装の段で書き、主張は作者が確かめる（WP-5R の文言もこの確認の対象に含める）。

五つの柱を置く。

1. **再生ではなく演奏。** 楽器そのものが browser で動き、その場で音を生む。server に音のファイルは無い。Start のたびに新しい seed で始まる（Manifesto「固定された録音物から生きた現象へ」、§9.3.129）。
2. **作品は台本。** 配るのは台本である。ページは演奏中の台本を楽器のすぐ下に見せ、そのまま持ち帰れるようにする。台本が決めるのは初期条件と区間の形で、一つ一つの音は生態が生む。同じ台本からは、同じ作品の別の演奏が生まれる。
3. **聴き手の模型が見える。** GUI は、聴き手の模型が聴いている粗さと調和の地形、Voice と Flora を示す。音楽理論の記号は出さない（既存の GUI の決まり）。
4. **正直な表示。** 各曲について、作曲者が書いたもの（初期条件、区間の形、聴き手の評価の設定）と、生態が生むものを分けて示す。人が書いた和声があれば冒頭で示す（§9.3.104 と警告の決まり）。楽器の版と commit も示す。性能の数値を送っていることと、その中身も示す。
5. **版に固定。** 演奏の場の engine、GUI、台本は、一つの release の tag から作る。

しないこと（非目標）は次のとおり。

- 録音の配布、演奏の保存・共有、音の書き出し。
- 利用者の登録、cookie、外部の解析 script、外部の font や CDN（演奏の場の page に限る。§5.6）。
- 聴き手が台本を編集して実行する機能（playground）、作品の投稿や共有。後者は将来の SNS の範囲である（§11）。live coding とも呼ばない（既存の位置づけ）。
- endless の作品。engine の入口が拒否する。Manifesto の約束としては残す。
- 共有世界、遠隔の参加、音声の入力、生体信号（§9.3.129 で別の候補とした）。
- モバイル端末への最適化（WP-0 の範囲外を引き継ぐ）。

## 3. 置き場所は conchordal.org/demo/（推奨で決めた）

conchordal.org の `/demo/` に置き、org の home と navigation からつなぐ（日本語の navigation も英語の `/demo/` へつなぐ）。理由は四つある。

- **conchordal.space は SNS のために空ける。** 今デモを置くと、SNS の設計の前に、その root、URL の形、privacy の形（cookie を持たず、受け取るのは性能の数値だけの場）が決まってしまう。SNS には登録や投稿が要るので、デモの場で約束した privacy の形と食い違う。
- **既存の試作がそのまま使える。** WP-5／WP-5R は org の Astro の中に作られている。
- **配信の条件が足りる。** org の zone は Cloudflare にあるので、GitHub Pages の前で response header（CSP、Permissions-Policy）を Cloudflare の Transform Rule で足せる。圧縮も Cloudflare が行う。性能の数値の受け口は、同じ zone の一つの path に Cloudflare の Worker を置けば足りる（§5.6）。ほかの path は静的な GitHub Pages のままである。
- **配備の周期は Release の asset で分けられる**（§5.7）。org の配備は文書が変わるたびに走るが、演奏の場の build は版ごとに一度だけ行い、org の配備はその成果物を取り込むだけにする。

戻し方：公開の前なら、path の変更や別の host への移動に費用はかからない。公開の後に移す時は、移した先へ redirect する。

## 4. 要件

番号は設計と検査から引く。厳しい数には、守る消費者か出どころを書く。

### 4.1 機能

- **F1** 開くと、曲の題、短い説明、音量の注意、Start が一画面に見える。読み込みの進みを示す。
- **F2** Start で新しい演奏が始まる。seed は毎回 host の乱数から作る。`?seed=` は調べもののために残すが、画面には出さない。
- **F3** 止める操作は一つにする（作者 2026-11-17）。「止める」は約 80 ms で音を絞って止め、最初の画面に戻る。WP-0 の End（穏便な終わり）は page では使わず、browser の GUI でも End を出さない。曲の自然な終わりは台本の release と fade に従う。
- **F4** 演奏中は、経過の時間と状態（準備中、演奏中、終わりに向かっている、終わった）を DOM の文字でも示す（screen reader 向けに aria-live）。
- **F5** GUI（地形の表示）は既定で見せ、隠すこともできる。隠しても音は続く。全画面にできる。
- **F6** 自然な終わりの後は、この演奏が終わったことを示し、新しい演奏を始められるようにする。展示のための繰り返しは残す（WP-5 の Repeat と `?repeat=1`）。
- **F7** 曲の台本を、GUI のすぐ下に、折りたたまずに読み取り専用で見せる。download でき、Rhai の本と GitHub の source へのリンクを付ける。編集の欄にはしない（live coding と誤解させないため。既存の位置づけ）。
- **F8** 曲が複数ある時は選べる。切り替えでは、前の演奏が止まってから次を起動する（WP-5 の直列化）。
- **F9** 英語だけにする（2026-11-20、作者「日本語はいらないな、英語ページのみでいいかも。Demo として」）。page の題は Demo、言語の切り替えと hreflang は出さない。目録 `pieces.json` も英語だけ。
- **F10** 対応しない browser では、足りないもの（WebAssembly SIMD、AudioWorklet、WebGL2）を示す。録音で代わりにしない。
- **F11** 楽器の版と commit、各曲の「作曲者が書いたもの／生態が生むもの」、第三者の license、privacy の説明を載せる。
- **F12** 演奏の場の asset を含む build では、org の home と navigation から演奏の場へリンクし、sitemap に載せる。asset の無い build には、page も link も無い（§12）。
- **F13** 性能の数値を送ることを、Start の近くに一行で示す。送る項目とその値を、演奏中にそのまま見られるようにする。送信は一回の操作で止められる（§5.6）。

### 4.2 品質

- **N1 音切れ 0。** 宣言した機械（N2）で、載せる各曲の全長を、GUI を見せた前面の tab で鳴らす。underrun 0、欠けた frame 0、生成と消費の frame 数の一致を確かめる（§9.3.130 の受入、WP-6 の手順）。
- **N2 宣言する機械と browser。** blanche（Linux の Chrome、Firefox）と、作者の M5 MacBook Air（Chrome、Safari、Firefox）。
  - 消費者：作者は「M5 はかなり新しいので、もうちょっと余裕があったほうがいい」と述べた（OP-1）。
  - 余裕は N3 の先行量で受ける。見込みでは、M5 の約 2.9 倍遅い機械まで、M5 で見た遅れに耐える。
  - Windows は測る機械が無いので、宣言に入れない。
- **N3 先行量の既定は 2000 ms。**
  - 出どころ：WA-1 の当座の手当て（`?buffer=2000`）。記録には「音の始まりが約 2 秒遅れるだけで、この曲には影響しない」とある。
  - 計算：一 hop の平均の、予算に対する比を m とする。機械の側の遅れが s 倍で T 秒続くと、s·m > 1 の時に先行は (s·m − 1)·T だけ減る。M5 で見た s = 1.7、T = 10 秒に対して、2 秒の先行がもつのは m ≤ (1 + 2/10)/1.7 ≈ 0.706 の時、すなわち平均 7.5 ms 以下の時である。
  - OP-3 の後の M5 の平均は、blanche の 2.25 ms に改善前の比（6.33/5.57 ≈ 1.14）を掛けて約 2.6 ms と見込む。未測定である。7.5/2.6 ≈ 2.9 が N2 の余裕になる。
  - m は曲の最も重い 30 秒の窓で見る（WA-1 では全体の平均の約 1.12 倍だった）。
- **N4 音切れが出た後の手当て。**
  - underrun の後は先行の目標を倍にし、worker は目標まで作り足す。worklet は欠けた所を無音で埋めて続ける今の作りのままで、再生は止めない。WP-0 で決めた「自動の対策」に当たる。
  - 上限は 8 秒とする（§12）。「止める」は出力の段で音を絞るので、先行の遅れを待たない。
  - 平均が予算を越え続ける機械では、追いつけないことを一度だけ知らせる。
- **N5 背景の tab でも続く。** worker は worklet の消費の知らせで動き、timer に頼らない（今の作り）。Chrome は音を出している tab を timer の強い throttling から外す。GUI は見えない間は描かない（requestAnimationFrame が止まる）。Safari と Firefox の振る舞いは受入の時に確かめる。
- **N6 読み込み。** engine と GUI の wasm は合計約 11.3 MB、gzip で約 3.5 MB（WP-5）。圧縮して配り、streaming で compile する。版ごとの path に置くので、中身は変わらず、cache の再検証（GitHub Pages の max-age 600 秒と ETag）で足りる。GUI の wasm は engine と並行に取る。
- **N7 Air-Gap と privacy。**
  - 音、音から作った値、演奏の結果を、保存・送信の API（download、upload、storage、clipboard、network）へ渡さない。
  - 外へ送るのは、明示した項目の性能の数値だけである（§5.6。計測ページの allowlist と同じ方針）。
  - 演奏の場の page は cookie を使わない。local storage には、送信を止めた設定の一つだけを置く。
  - 外の origin へは一つも request を出さない（font、解析、CDN を含む。今の BaseLayout は外部の資源を読み込まない）。数値の受け口も同じ origin に置く。
  - server は静的な GitHub Pages である。POST を受けるのは数値の受け口一つだけで、受け口は IP と識別子を保存しない。
- **N8 安全な音量。** 既存の limiter と output gain をそのまま使う。Start の前に音量の注意を出す。
- **N9 keyboard の操作。** 開始、止める、GUI の表示、全画面、言語、曲の選択を keyboard で操作できる。

### 4.3 制約

- **C1** native の楽器と `conchordal-render` の挙動を変えない（WP-0 の条件）。engine の変更は main を通す。
- **C2** 楽器（`conchordal` binary）の Air-Gap を変えない。演奏の場のための build は、楽器の binary に disk への書き出しを足さない。
- **C3** 費用のかかる service を使わない（GitHub Pages、GitHub Actions、Cloudflare の無料の範囲）。
- **C4** 配備と公開は、毎回作者の許可を得て行う（外部への公開に当たる）。Release の asset の追加も Release の公開の一部として扱う。

## 5. 設計

### 5.1 全体

```text
release tag ──demo_assets.yml──> 演奏の場の asset（wasm・JS・台本・目録）──> GitHub Release に添付
main の web/ ──web_deploy.yml──> 固定した版の asset を取り込み ──> Astro の build ──> GitHub Pages（Cloudflare 経由）

聴き手の browser
  main thread                       Web Worker                      AudioWorklet
  /demo/ の page + egui GUI <-- snapshot --  StepEngine を一 hop ずつ  -- PCM --> 再生
  Start / Stop ------------------- 命令 ----->  (worker → 生態の解析 → 聴き手の解析)
        |
        +-- 性能の数値（演奏ごとに一件、明示した項目だけ）--> /demo/metrics（Cloudflare Worker）
```

engine と音の経路は今の `crates/conchordal-web` をそのまま使う。新しく作るのは、player の小さな変更、asset の組み立てと Release への添付、org の配備での取り込み、数値の受け口である。page は WP-5／WP-5R を移す。

### 5.2 page

WP-5／WP-5R の `web/src/pages/demo.astro`、`web/src/components/DemoPage.astro`、`web/src/scripts/demo.js` を基盤にする（BaseLayout、Navigation、Footer を使う）。

画面は上から次の順に並べる。2026-11-20 に作者の「デモページをもっと簡素に」「ページのレイアウト洗練させて」で整えた。英語だけの page で、題は Demo。

1. 題（Demo）と版。その下に曲目のリスト（一行に一曲、曲名と長さ、選んだ曲に印）。曲が増えたら行を足す（2026-11-20、作者「Line, Rain が曲名のようにみえない。これから増やしてゆくので、リスト形式がいいかな？」）。
2. 演奏の panel を一つ。上の一列に、Start と Stop、状態と経過時間、右に Repeat、Room、Display、Full screen。音量の注意は準備完了の状態の表示に含める。
3. GUI の canvas。既定で表示する。1200×900 の基準 layout を区画に収める縮尺は、WP-5 の `gui/src/browser.rs` の変更を移す。panel の下端に、性能の数値を送る切り替え（Send performance counters）と「No sound is sent」、詳細への link（F13）、対象の browser と読み込みの量を一行で置く。
4. その下に二段：左に台本（読み取り専用、折りたたまない、Download と GitHub の link。F7、§5.10）。台本の題の下に「Every Start is a new performance of the same score.」を添え、自然に終わった時の状態の表示は「Ended. Start again for a new performance.」にする（2026-11-20、作者「演奏するたびに違うということをさりげなく強調したい」。seed は Start ごとに新しく引く）。右に曲の題と、人が書いた和声がある時の警告、作曲者が書いたもの／生態が生むもの。台本と、それが何を決め何を生態に任せたかを並べて読める。幅の狭い画面では一段。
5. 性能カウンターの詳細（送る項目、送らないもの、止め方、保持の期間、いまの演奏の記録）を折りたたんで置く。版と commit、license は page の末尾。到達点の表は外した（technote §9.1 が正本）。

状態の遷移は次のとおり。

- `unsupported`：F10 の表示で止まる。
- `idle` → Start → `loading`（wasm の取得と compile。初回だけ）→ `priming`（先行量が満ちるまで作る）→ `playing`。
- `playing` → 自然な終わり → `ended`。
- どの状態からでも「止める」 → `idle`。
- `error`：読み込みの失敗や台本の拒否。台本の拒否は asset の組み立ての検査（§6）で防ぐ。

### 5.3 曲の目録

曲は `samples/demo/` に置き、目録 `samples/demo/pieces.json` を唯一の編集元にする（WP-5 の `web/src/data/demo-scores.json` から移す）。台本が版の tree に入るので、asset は tag から一意に組める。native の利用者も同じ台本を手に取れる。

```json
[
  {
    "id": "example",
    "title": { "en": "…", "ja": "…" },
    "score": "example.rhai",
    "config": "example.toml",
    "duration_sec": 354,
    "authored": { "en": "…", "ja": "…" },
    "generated": { "en": "…", "ja": "…" },
    "authored_harmony": null
  }
]
```

- `config` は省略でき、省略すれば既定の設定で動く。
- `authored_harmony` は、人が書いた和声がある時に、何を誰が書いたかを書く。null なら無い。
- 台本は seed を固定しない。top-level の sample と同じ決まりで、`tests/sample_seed_policy.rs` の対象に加える。
- 台本は `tests/compile_only_samples.rs` の対象にも加える。
- 目録の項目は org の layout に依らない形にする（§11 の SNS が作品の記述として再利用できるように）。
- 掲載は作者が採った曲だけにする。

### 5.4 音の経路と先行量

今の経路（Worker の `StepEngine` → transferable の PCM → AudioWorklet、snapshot → GUI）をそのまま使う。player で変えるのは次の三点である。

- `bufferMs` の上限（今は 2000）を外し、既定を 2000 にする（N3）。
- worklet から underrun の知らせを受けたら、worker の `targetFrames` を倍にする（N4）。worker の pump は目標まで作り足す作りなので、ほかの変更は要らない。
- 状態（とくに `ending`）を page へ知らせる。

性能の数値は、今の player が集める値（`result` と worker の metrics）から、§5.6.1 の項目だけを page が組み立てて送る。計測ページ（`crates/conchordal-web/www/`）の `page.js` は演奏の場に載せない（WP-5 と同じ）。

### 5.5 GUI

- 既定で表示する。聴き手の模型が聴いている地形を見せることが、録音との違いを示す（柱 3）。
- 「表示しない」を選ぶと canvas を外し、main thread の描画の負荷を除く。snapshot は欠かさず順に渡す必要がある（符号化の状態が続く）。そのため、表示しない間も decode を続けるか、表示を戻す時に GUI を reset して最新の snapshot から作り直すかのどちらかにする。費用の小さい方を実装の段で選ぶ。
- GUI の中の Start／Exit と page の操作は、同じ player を呼ぶ（WP-5 で確かめた形）。GUI の Exit は「止める」と同じ操作である。

### 5.6 Air-Gap と privacy の担保

文書の約束だけでなく、browser と server の設定で閉じる。

- **page の CSP（`<meta http-equiv>`）。** 演奏の場の page に次を置く。GitHub Pages は header を設定できないので、まず meta で掛ける。
  - `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self'; style-src 'self'; img-src 'self'; form-action 'none'; base-uri 'none'`
  - meta の CSP は document にだけ効き、Worker の script の中の通信は縛らない。また `frame-ancestors` と Permissions-Policy は meta では掛からない。
- **Cloudflare の Transform Rule（作者の一度だけの作業）。** `/demo/*`、`/demo-assets/*` の response に、同じ CSP と `frame-ancestors 'none'`、`Permissions-Policy: microphone=(), camera=(), geolocation=()`、`Referrer-Policy: no-referrer` を足す。これで Worker と AudioWorklet の script も同じ方針で縛られる。
- **server。** GitHub Pages は静的な file だけを返す。POST を受けるのは、§5.6.1 の数値の受け口だけである。
- **検査（§6）。** wasm の import の一覧（WP-5 の方法）、一曲の全長の間の network の記録（同じ origin の GET と、数値の受け口への明示した項目の POST だけであること）、CSP の違反 0。
- **説明。** Cloudflare と GitHub は IP などを配信と security のために扱う。privacy の説明にそう書く。訪問の数は Cloudflare の server 側の集計で見て、client の解析 script は置かない。

#### 5.6.1 性能の数値の送信

目的は性能の改善である。宣言した機械の外で音切れが起きた時に、作者が知らないまま放置しないため（WP-0 で作者が提案した仕組み）。

- **数値が答える問い。** 送る項目は、次の三つの判断に要るものに限る。
  1. 宣言の外の機械で、音切れがどれだけ起きているか。
  2. 起きた音切れは、機械が平均で遅いためか、一時の遅れ（他の仕事、熱、電源の状態、効率の core への移動、tab の裏への移動）のためか、曲の重い区間のためか。これで、手当てが engine の高速化か、先行量か、曲かが決まる。
  3. 次の高速化を、どの段（worker、生態の解析、聴き手の解析、snapshot、GUI）に向けるか。
- **ハードの情報について。**
  - CPU の型名とクロックを返す web の API は無い。User-Agent も型名を含まない。2026-11-15 の M5 MacBook Air（Chrome 157）の記録は `Intel Mac OS X 10_15_7`、論理 core 10、memory 32 で、送っていたハードの情報からは M5 と分からなかった（`.orchestration/units/WA-1/metrics.jsonl`）。
  - クロックや型名は一 hop の時間を決めない。同じ M5 でも約 10 秒、約 1.7 倍遅くなった（OP-1）。熱、電源、core の割り当て、browser の engine、背景の負荷が効く。
  - 問い 1〜3 に答えるのは、その機械で実際に測った時間の列と、基準機との比である。ハードの情報は、問題を機械の種類に結び付けて手元で再現する時と、必要な機械の条件を公に書く時に要る。そこで既定では送らず、音切れが起きた時だけ本人に尋ねる（下の「ハードの情報の追加」）。
- **送る項目**（allowlist。計測ページ `www/page.js` の項目から選ぶ）：
  - 演奏：曲の id、engine の版と commit、状態の遷移（終わり方：自然、止める、曲の切り替え、tab を閉じた）、演奏の長さ。
  - 音の経路：sample rate（予算は sample rate で変わる）、AudioContext の `baseLatency` と `outputLatency`、先行の目標と最小の残量、生成・消費・欠けた frame 数、underrun の回数と、演奏の始めからの時刻。
  - 計算の全体：hop の時間の分布（平均、p50、p95、p99、最大）、snapshot と GUI の時間の分布、GUI の表示の有無、tab が裏にあった時間。
  - **5 秒の窓ごとの列**：hop の時間の平均と最大、先行の最小の残量、underrun の数、tab の表裏、GUI の表示。問い 2 を分けるのはこの列である。OP-1 も、計測ページの 5 秒ごとの記録の列から一時の遅れを読んだ。6 分の曲で約 70 窓、数 KB になる。
  - **段ごとの時間**：一 hop の中の worker、生態の解析、聴き手の解析の時間の平均と p99（問い 3）。`StepEngine` に段の時刻を足す。web の build だけの計測で、出力は変えない。
  - 環境：browser と OS の種類と版（request の header の User-Agent から受け口で粗くまとめる）。
- **送らない項目**：音、snapshot、波形、peak、非零 sample 数など音から作った値、台本、error の本文（error は種類だけ）、識別子、ハードの情報（下の追加を本人が選んだ時を除く）。
  - 端末の情報は、計測ページでは送っていたが、既定では除く。通知だけで既定で送る形にするので、端末から読み取る情報を最小にする（EU の ePrivacy の同意の対象になりうるため）。
- **速さの比。** 公開の前に、載せる各曲を基準機（blanche の Chrome）で複数の seed で鳴らし、窓ごとの hop の時間の中央値と、seed による揺れの幅を取る。届いた列を同じ曲・同じ窓の中央値で割った比が、その機械の速さの目安になる。比が演奏の全体で高ければ機械が遅い。一部の窓だけ高ければ一時の遅れか曲の重い区間で、基準の列と比べて分ける。seed の揺れが大きくて比が読めない時に限って、決まった仕事を測る校正の段を足すことを考える。
- **ハードの情報の追加（本人が選んだ時だけ）。** 音切れが起きた演奏の終わりに、「改善のため、この機械の情報も送りますか」と尋ね、送る値を見せる。
  - 項目：論理 core の数、memory（Chromium だけ、丸めた値）、CPU の architecture（User-Agent Client Hints、Chromium だけ）、GPU の名前（WebGL の renderer の文字列。Chromium の Apple Silicon では SoC の名前が出ることが多い。確かめていない）。
  - 送るのはその演奏の記録に付ける一回だけで、次の演奏には付けない。
- **送る時**：演奏ごとに一件で、窓ごとの列を含める。自然な終わり、「止める」、曲の切り替え、tab を閉じる時（`pagehide` で `navigator.sendBeacon`）。計測ページの 5 秒ごとの送信は、ここでは使わない。
- **止め方**：Start の下の切り替えで止める。止めた設定は local storage の一つの key に置き、次に開いた時も止まったままにする。browser が Global Privacy Control（`navigator.globalPrivacyControl`）を示す時は、既定で送らない。
- **見せ方**：「送る中身」を開くと、いまの演奏の項目と値が、送るものと同じ形で並ぶ（F13）。
- **受け口**：`https://www.conchordal.org/demo/metrics` に Cloudflare の Worker を route で置く（ほかの path は GitHub Pages）。同じ origin なので、CSP の `connect-src 'self'` のままでよい。
  - Worker は allowlist の外の項目と大きすぎる body を捨てる。IP は保存しない。
  - 保存先は、保持の期間に上限があり、統括が読み出せるもの（Workers Analytics Engine または D1。実装の段で無料の範囲を確かめて選ぶ）。
- **計測ページとの違い**：計測ページ（`www/page.js`）は手元の測定のための道具なので、今の項目と 5 秒ごとの送信をそのまま残す。
- org の home（TopLayout）は Google Fonts を読み込む。演奏の場の page は BaseLayout なので、この読み込みは無い。home の扱いはこの設計の範囲外とする。

### 5.7 build と配信

**asset の組み立て（WP-5 の `web/scripts/build-demo.mjs` を改める）。**
1. `crates/conchordal-web/build.sh` で、engine と GUI の wasm と JS の glue を作る（既存の手順と版。wasm-bindgen-cli 0.2.130、`+simd128`）。
2. wasm の name 区画を除く（WP-5 の方法）。
3. `demo-assets/<tag>-<commit8>/` に、wasm、glue、共通の JS（`player.js`、`engine-worker.js`、`audio-worklet.js`）、`samples/demo/` の目録と台本と設定、`build.json`（tag、commit、engine の版）を置く。
4. 第三者の license の一覧を作り、同じ場所の `licenses.txt` に置く。wasm に含まれる crate の MIT／Apache の通知を配るためである。道具は実装の段で選ぶ。
5. smoke の検査（WA-1 の smoke と WP-5 の操作の検査を `web/scripts/demo-smoke.mjs` に移したもの）を通す。

**Release への添付（新しい workflow `.github/workflows/demo_assets.yml`）。**
- 起動：release の公開（`release: published`）と手動（`workflow_dispatch`、tag を指定）。
- 手順：tag の checkout、Rust（WP-5 の 1.99.0）、`wasm32-unknown-unknown`、wasm-bindgen-cli 0.2.130、Node、asset の組み立てと smoke、`conchordal-demo-<tag>.tar.gz` を Release に添付、`web_deploy.yml` を起動。
- 秘密は要らない（`GITHUB_TOKEN` の `contents: write` と `actions: write`）。

**org の配備での取り込み（`web_deploy.yml` を改める）。**
- 演奏の場の版は main の `web/src/data/demo-release.json`（`{"tag": "v…"}`）で固定する。版を替えるのはこの一行の commit で、何が配られているかが履歴に残る。
- Astro の build の前に、その tag の Release から `conchordal-demo-<tag>.tar.gz` を `gh release download` で取り、`web/static/demo-assets/` に展開する。
- asset がまだ無い時（tag を打った直後など）は、演奏の場の page を作らず、navigation にも出さない。文書の配備は止めない。
- 曲の題と説明は、展開した目録から build の時に読む（静的な HTML になる）。
- WP-5 が `web_deploy.yml` に入れた wasm の build は採らない。

**数値の受け口。**
- Worker の source と `wrangler.jsonc` を `web/metrics-worker/` に置く。route は `www.conchordal.org/demo/metrics`。
- Worker は小さく、変えることがまれなので、配備は wrangler で手で行う（作者の Cloudflare の login が要る）。CI に Cloudflare の秘密を置かない。
- 受け口が無い時や止まっている時でも、演奏は変わらない（`sendBeacon` の失敗は無視する）。

### 5.8 版との結び付け

- 演奏の場は `demo-release.json` が指す一つの release の版で動く。page の版の表示は tag と commit である。
- 古い版を `/demo/v/<tag>/` に残すことは、asset が各 Release に残るので安くできる。最初の公開でするかは後で決める（§8）。
- 載せる曲が要る engine の機能は、その版の main に入っていなければならない。
  - 2026-11-19 の統合（台帳 §9.3.142）で、`respond_to_onsets`、雨の語、感じられた拍の研究の設定を、既定 0 のまま一本にまとめた。台帳の節番号も付け直した（`respond_to_onsets` は §9.3.138）。
  - 聴き手の慣れは、同じ日に既定が形 P（`drive = "channel_p"`）になった（台帳 §9.3.140）。載せる二曲は既定で鳴らす。曲ごとの設定は `pieces.json` の `config` で選べる。
- 送り手の音型（GT）に頼る曲は、まだ載せられない。GT は研究の branch に残っている。

### 5.9 WP-5／WP-5R から移すもの、改めるもの

- 移すもの：page と component、`demo.js`（直列化、全画面、canvas の wheel）、`gui/src/browser.rs` の縮尺、`build-demo.mjs`、到達点の表、WP-5R の design token と構成。
- 改めるもの：曲の目録の置き場所（§5.3）、Release の asset による版の固定（§5.7）、先行量（§5.4）、GUI の表示の切り替え（§5.5）、CSP（§5.6）、公開の時の noindex の解除と navigation（F12）。
- worktree は `317e7931` が基準なので、main から新しい branch を作って移す。worktree の未 commit の内容は、移し終えるまで消さない。git に入っていない内容なので、消すのは作者が確かめた後にする。

### 5.10 台本の表示

- GUI のすぐ下に、演奏中の曲の台本の全文を読み取り専用で置く。折りたたまない。最初の画面は楽器と Start のままで、scroll すると台本が見える。
- 見せる理由：台本が決めるのは初期条件と区間の形だけで、一つ一つの音は生態が生む。短い台本と、そこから生まれる演奏を並べて見せることが、この楽器の説明になる。alpha の読み手（作曲と開発の人）が、自分で書く時の手本にもなる。
- 隣に並べる形（GUI の横に編集の欄を置く形）は採らない。GUI の基準の幅（1200）を削り、live coding の道具と誤解させるからである。
- 等幅の文字で見せる。色付けは必須にしない。付ける場合も、外の CDN を使わず、page の中の小さな処理で行う。
- 演奏中に、いま効いている区間を台本の上で示すことは、後の候補とする。台本の位置付きの診断の仕組み（`scripting/context.rs`）を使える見込みだが、確かめていない。

## 6. 検査と受入

実装の検査、実機の受入、作者の判断を分けて記録する。

**実装の検査**（統括と worker が行う）：
- engine や GUI の Rust を変えた時は、root の fmt、clippy、test。page と player だけの変更では、JS の構文と動作の検査、org の build に留める。
- smoke：載せる各曲を headless Chrome で 20 秒、音を出さずに鳴らす。例外 0、underrun 0、台本の受理を確かめる。
- Air-Gap：§5.6 の三つの検査。
- 配備の経路：asset の無い状態と有る状態で、org の build が通り、page の有無が切り替わること。
- 数値の送信：送る body が allowlist の項目だけであること、「送る中身」の表示と送る body が一致すること、止めた時と Global Privacy Control の時に POST が 0 であること、受け口が allowlist の外の項目を捨てること。
- 操作：開始、止める、繰り返し、曲の切り替え、全画面、GUI の表示の切り替え、背景の tab、両言語。1200×900 と 1440×900 の画面で、R/H と GUI の Exit が見えること（既存の GUI の要件）。

**実機の受入**（N1、N2）：
- blanche の Chrome と Firefox で、各曲の全長を、GUI を見せた前面の tab で鳴らす。
- 作者の M5 MacBook Air の Chrome、Safari、Firefox で同じことを行う（作者の手元で）。あわせて Chrome で一度、背景の tab のまま全長を鳴らす。
- 記録する項目は WP-6 と同じ（先行の最小、hop の平均・p99・最大、GUI の時間、browser、負荷）。

**作者の判断**：
- 載せる曲の採択（作品の約束、§9.3.129）。
- 演奏の場の文言（作者名義の文章）の確認。
- 配備と公開の許可。

## 7. 次版に入れる作業（依存順）

1. **S-1** WP-5／WP-5R を main の上へ移し、player の三点の変更、GUI の表示の切り替え、台本の表示、page の CSP を入れる。依存なし。
2. **S-2** asset の組み立て（目録の `samples/demo/` への移動、`build.json`、license の一覧）と smoke の検査。S-1 の後。
3. **S-3** `demo_assets.yml`、`web_deploy.yml` の取り込み、`demo-release.json`。S-2 の後。
4. **S-3b** 性能の数値の送信（page の側の表示と切り替え、窓ごとの列、`StepEngine` の段ごとの時刻、音切れの後のハードの情報の問い、受け口の Worker、保存先）と、載せる曲の基準機での基準の列。S-1 の後。配備は作者の login と許可の後。
5. **S-4** 曲が要る機能の main への統合。作者が曲を選んだ後（2026-11-20 に選んだ）。
6. **S-5** 実機の受入。S-1〜S-4 の後。M5 は作者の手元で行う。
7. **S-6** 公開。`demo-release.json` の tag を、asset の付いた release に書き換える（link と sitemap は asset に従って出る）。作者の許可の後。

S-1〜S-3b は、曲が決まる前に、main で動く仮の台本で進められる。

## 8. 作者の判断

**いま決める：0 件。**

**別の選択で自動的に決まる：**
- main に入れる研究の機能（§5.8）は、載せる二曲で決まった（苔の `respond_to_onsets`、雨の `follow_meter`、部屋の第 1 段階）。

**後で足りる（公開の前に）：**

1. 古い版を残すか（`/demo/v/<tag>/`）。
   - 推奨：最初の公開では残さない。作品の継承（§9.3.129 の経路の 4）を実例で確かめる時に開く。asset が各 Release に残るので、後から足しても費用は小さい。

**作者が決めたこと（2026-11-20）：**
- 載せる曲は「雨と苔（研究）」と「線と苔（鋭い応答、研究）」の二曲（作者「2分になってるもの二件を採用」）。部屋の響きを入れた page で六曲を聴き直した後の答えである。ほかの四曲（B2、拍を聴く雨の B2、線と苔、慣れの形 P の線と苔）は目録と `samples/demo/` から外した。台本は `3180847b` に残る。
- 続けて作者は、二曲の順を逆に（「線と苔」が先）、曲名を簡素に（「線と苔」「雨と苔」）とし、これまで印象がよかったものから協和のデモと、できれば temporal DCC のデモを二、三足したいと述べた。候補として sample 13（「三つの窓」。2026-10-12 に E-2 の四曲が「どれもまえのsamples より格段に音楽的」と評され、13 は直した版より元の形が好まれた）と sample 08（「群れ」。作者は「徐々に歩調をあわせてゆくようにきこえる」と述べた）を、台本の code を変えずに目録へ足した。作者は聴いて「たした二つの曲は、基準に足っしてないのでやめる」と答え、二曲とも外した。載せる曲は「線と苔」と「雨と苔」の二曲のまま。
- 曲名は作者が推奨を採り、Line と Rain にした（2026-11-20、両言語とも英語）。苔は両方の曲にいるので、題は二曲の違いを指す。雨は平均 0.5 滴/秒から最初の 46 秒で 6 滴/秒へ増やして入り、その後は 1/f の揺れだけで変わる（作者「最初はポツポツからはじめてだんだん増やしていきたい」）。

**作者が決めたこと（2026-11-16）：**
- 性能の改善のため、性能の数値を送る。送っていることを表示する（作者「性能改善のため、パフォーマンスカウンタを仕込んでることは表示したほうがよいね」）。送り方の細部（§5.6.1）は推奨で決めた（§9）。

**作者の一度だけの作業（判断ではない）：**
- Cloudflare の dashboard で、§5.6 の Transform Rule（response header）を org の zone に置く。公開の前に要る。
- 数値の受け口の Worker を wrangler で配備する時に、Cloudflare に login する（配備そのものも作者の許可で行う）。

**決まれば動くこと：** 曲が決まったので、S-4（main への統合）と、その後の S-5 に着手できる。

## 9. 推奨で決めたこと（違えば戻せる）

- 置き場所は conchordal.org の `/demo/`（§3、英語だけ）。conchordal.space は SNS のために使わない。
- 演奏の場の build は release の tag から行い、Release の asset を org の配備が取り込む。版は `demo-release.json` で固定する。
- 基盤は WP-5／WP-5R の試作。
- 曲の目録と台本は `samples/demo/` に置く。
- GUI は既定で表示し、隠せるようにする。
- 台本は GUI の下に、折りたたまずに読み取り専用で見せる。GUI の横に編集の欄を置く形は採らない（§5.10）。
- 性能の数値は、通知して既定で送り、一回の操作で止められるようにする。Global Privacy Control を示す browser には既定で送らない。止めた設定だけを local storage に置く。
- 送るのは演奏ごとに一件で、5 秒の窓ごとの列と段ごとの時間を含める。ハードの情報（core の数、memory、architecture、GPU の名前）は既定で送らず、音切れが起きた演奏の後に本人が選んだ時だけ付ける。browser と OS は受け口が User-Agent から粗くまとめる。
- 速さは、基準機で複数の seed で取った同じ曲・同じ窓の中央値との比で読む。
- 受け口は `www.conchordal.org/demo/metrics` の Cloudflare の Worker とし、配備は wrangler で手で行う（CI に秘密を置かない）。
- sample 01〜16 は載せない（2026-11-20 に 13 と 08 を候補として試し、作者が基準に達していないと答えた）。playground は入れない。
- seed は画面に出さない。
- モバイルでは止めずに、desktop の browser 向けであることを知らせる。
- 先行の既定は 2000 ms とし、underrun の後は倍にする。

## 10. 確かめていないこと

- Cloudflare が GitHub Pages から来る `application/wasm` を圧縮して配るか。配備の後に `Accept-Encoding: br, gzip` を付けて確かめる。圧縮されなければ、Cloudflare の圧縮の規則を足す。
- OP-3 の後の M5 の実測。N3 の 2.6 ms は見込みである。
- Safari での動作。Firefox は DEMO-B1 で Chrome の約 1.37 倍遅かった（WA-1）。
- 長い時間、背景の tab に置いた時の snapshot の溜まり方。
- §5.6 の CSP で、eframe と wasm-bindgen の glue が動くか（違反の数で確かめる）。
- `web_deploy.yml` の中で `GITHUB_TOKEN` を使って Release の asset を取れるか（public の repository なので取れる見込み）。
- Cloudflare の Worker の route（`/demo/metrics`）が、GitHub Pages を origin とする proxy と並んで動くか。保存先の無料の範囲と保持の期間。
- 端末の情報を除き、通知だけで既定で送る形が、EU の ePrivacy の扱いで足りるか（法的な確認はしていない）。
- 基準機の複数の seed で、窓ごとの hop の時間の揺れが、速さの比を読める幅に収まるか（公開の前の基準の取得で分かる）。
- WebGL の renderer の文字列が、各 browser で何を返すか（Chromium の Apple Silicon で SoC の名前が出るか、Safari と Firefox で丸められるか）。

## 11. 将来の conchordal.space（作品を共有する SNS）との関係

作者の予定（2026-11-16）では、conchordal.space は将来、作品を共有する SNS になる。この設計ではその SNS を設計しない。ただし、次の二点で道を塞がない。

- **作品の単位を共通にする。** この場の単位は「一つの作品（台本、設定、作曲者が書いたもの／生態が生むもの、人が書いた和声の有無）を、聴き手の手元で一回ずつ演奏する」ことである。作品を台本で共有する SNS の作品の page も、同じ単位で作れる。目録の項目（§5.3）と player は org の layout に依らない形にし、SNS が取り込めるようにする。
- **conchordal.space を今は使わない。** root、URL の形、privacy の形は SNS の設計で決める。

SNS の設計で別に決めることを、気づいた範囲で残す。
- 登録、投稿、moderation、privacy の形。演奏の場の「cookie を持たず、受け取るのは性能の数値だけ」という約束は、SNS には引き継げない。
- 他人の台本を聴き手の browser で動かす時の制限（Rhai の実行量、memory、endless の扱い）。
- §9.3.129 の共有世界の候補（一つの生態系を多数で聴く形）との関係。

## 12. 実装の段で決めたこと（2026-11-16）

- **先行量の上限 8 秒。** 平均で実時間に間に合う機械（m ≤ 1）が、M5 で見た遅れ（s = 1.7、T = 10 秒）を越えるには、(1.7 − 1) × 10 = 7 秒の先行が要る。2 秒からの倍化で 7 秒を超える最小の段が 8 秒である。m > 1 の機械は先行をいくら増やしても追いつかない。
- **CSP は inline の script と style を許す。** org の layout は小さな bundle（navigation の開閉の script）を inline で出し、演奏の場の theme も inline の style である。CSP の役目は外の origin への request を塞ぐことで、`default-src`、`connect-src`、`img-src`、`font-src` の `'self'` がそれを保つ。build した page で、違反 0、外の origin への request 0 を smoke の検査で確かめる。
- **保存先は Workers Analytics Engine。** 無料の plan で一日 10 万件まで書け、一件の文字列は 16 KB まで、保持は 3 か月で自動に消える（2026-11-16 に Cloudflare の文書で確認）。一回の演奏の記録は約 1 KB（20 秒）で、6 分の曲でも数 KB に収まる。
- **公開の切り替えは `web/src/data/demo-release.json` の tag 一つ。** route `[demo].astro` は、asset がある時だけ `/demo/` を作る。navigation の link と sitemap も同じ条件に従う。noindex を外す段は無くなった。tag を書き換えて main に入れることが公開に当たり、作者の許可で行う。
- **演奏する台本は page に見せている台本そのもの。** page は台本の全文を HTML に埋め込み、player はその文字列を engine に渡す。build した HTML から取り出した台本は、元の file と byte 単位で一致した。
- **残る小さな差。** asset の無い build でも、どこからも参照されない DemoPage の CSS と JS の bundle が `_astro/` に出る。ほかの page は main の build と byte 単位で一致した。

