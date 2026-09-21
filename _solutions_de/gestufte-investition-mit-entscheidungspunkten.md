---
title: Gestufte Investition mit Entscheidungspunkten
description: Finanzierung der Modernisierung in Tranchen, von denen jede
  Information erkauft, mit einer festgelegten Entscheidung an jedem Tor —
  einschließlich der Entscheidung, aufzuhören.
category:
- Management
- Business
- Process
problems:
- modernization-roi-justification-failure
- difficulty-quantifying-benefits
- modernization-strategy-paralysis
- history-of-failed-changes
- budget-overruns
- analysis-paralysis
- decision-paralysis
- incomplete-projects
- second-system-effect
- planning-credibility-issues
- system-stagnation
- inability-to-innovate
- poor-planning
- premature-technology-introduction
- technology-lock-in
- upgrade-blocked-by-customization
layout: solution
lang: de
en_slug: staged-investment-with-decision-gates
related_solutions:
- slug: no-regret-moves
  similarity: 0.7
- slug: pilot-projects
  similarity: 0.65
- slug: modernization-options-comparison
  similarity: 0.65
- slug: executive-sponsorship
  similarity: 0.65
- slug: total-cost-of-ownership-transparency
  similarity: 0.65
- slug: strangler-fig-pattern
  similarity: 0.65
---

## Description

Gestufte Investition finanziert eine Modernisierung in einer Sequenz kleiner Tranchen, von denen jede erwartungsgemäß Unsicherheit reduziert statt das gesamte Ergebnis zu liefern, und von denen jede an einem Tor endet, an dem eine festgelegte Entscheidung getroffen wird: fortsetzen, Ansatz ändern oder aufhören. Sie adressiert das strukturelle Problem großer Legacy-Vorschläge — die Organisation wird gebeten, eine große, unsichere Summe gegen einen Nutzen zu verpflichten, den sie nicht verifizieren kann, basierend auf einer Schätzung, die produziert wurde, als am wenigsten bekannt war. Diese Anfrage wird rational abgelehnt, weshalb so viele gut begründete Modernisierungsfälle scheitern. Stufung ändert, worum gebeten wird. Die erste Tranche bittet nicht um Genehmigung der Modernisierung; sie bittet um einen kleinen Betrag, um herauszufinden, was die Modernisierung tatsächlich kosten würde, mit einem expliziten Recht, sie danach aufzugeben.

## How to Apply ◆

> Eine Legacy-Modernisierungsschätzung, die erstellt wird, bevor jemand ein Stück davon versucht hat, ist keine Schätzung, und jeder im Raum weiß das — Stufung ist die ehrliche Antwort darauf.

- **Lassen Sie die erste Tranche Information erkaufen, nicht Fortschritt.** Extrahieren Sie eine kleine Fähigkeit, migrieren Sie eine Tabellengruppe, führen Sie einen Parallelvergleich durch. Das Liefergut ist eine verteidigbare Schätzung des Ganzen, produziert aus der Durchführung eines repräsentativen Stücks.
- **Bemessen Sie jede Tranche nach dem, was sich die Organisation zu verlieren leisten kann.** Der Test ist, ob der Sponsor sich damit wohlfühlen würde, sie vollständig abzuschreiben. Wenn nicht, ist sie zu groß, und das Tor wird nicht funktionieren, weil ein Stopp politisch unmöglich sein wird.
- **Definieren Sie die Torkriterien, bevor die Tranche beginnt**, schriftlich, einschließlich dessen, welches Ergebnis Stopp bedeuten würde. Ein Tor, dessen einzig mögliches Ergebnis "fortsetzen" ist, ist ein Meilenstein, und die Organisation wird es als solchen behandeln.
- **Machen Sie das Aufhören zu einem respektablen Ergebnis.** Das erste Mal, dass ein Tor genutzt wird, um etwas zu stoppen, ist es, was feststellt, ob der Mechanismus echt ist. Wenn Stoppen als Versagen behandelt wird, werden nachfolgende Tranchen Erfolg melden, unabhängig davon, was passiert ist.
- **Schätzen Sie den Rest bei jedem Tor neu**, unter Nutzung dessen, was die abgeschlossenen Tranchen tatsächlich gekostet haben, statt des ursprünglichen Plans. Legacy-Schätzungen verbessern sich dramatisch, sobald ein repräsentatives Stück erledigt wurde, und die überarbeitete Zahl ist das wertvollste Ergebnis der frühen Phasen.
- **Sequenzieren Sie so, dass jede Tranche etwas von Wert stehen lässt.** Eine Stufe, die nur bedeutsam ist, wenn die folgenden Stufen geschehen, erzeugt die Alles-oder-nichts-Verpflichtung innerhalb der gestuften Struktur neu.
- **Berichten Sie bei Toren in den Begriffen des Sponsors** — überarbeitete Kosten, überarbeiteter Nutzen, was gelernt wurde, was die Entscheidung ist — statt als technischer Fortschritt. Ein Torbericht, der architektonisches Wissen erfordert, um interpretiert zu werden, wird genehmigt werden, ohne verstanden zu werden, was den Mechanismus zunichtemacht.
- **Halten Sie die Tore selten genug, um bedeutsam zu sein.** Monatliche Tore werden zu Statusmeetings; vierteljährliche erzwingen eine echte Entscheidung. Das richtige Intervall ist ungefähr die Zeit, die die nächste Tranche braucht, um zu ändern, was bekannt ist.
- **Protokollieren Sie die aufgegebenen Optionen.** Eine Tranche, die feststellt, dass ein Ansatz nicht funktionieren wird, hat ein echtes Ergebnis produziert, und die Dokumentation davon verhindert, dass derselbe Ansatz in zwei Jahren erneut von jemandem vorgeschlagen wird, der nicht dabei war.

## Tradeoffs ⇄

> Stufung macht große Modernisierungen finanzierbar, indem sie die Verpflichtung begrenzt, auf Kosten von Overhead bei jedem Tor und dem Risiko, dass eine Anstrengung auf halbem Weg aus Gründen gestoppt wird, die nichts mit ihren Verdiensten zu tun haben.

**Vorteile:**

- Die anfängliche Bitte ist klein genug, um genehmigt zu werden, was häufig der Unterschied zwischen einer beginnenden Modernisierung und einer im vierten Jahr abgelehnten ist.
- Schätzungen verbessern sich schnell, weil jede Tranche Evidenz statt Analyse produziert — und Legacy-Kostenschätzungen ohne Evidenz sind fast wertlos.
- Die Exposition der Organisation ist an jedem Punkt begrenzt, was Sponsoren bereit macht, Arbeit mit echt unsicheren Ergebnissen zu unterstützen.
- Ansätze, die nicht funktionieren werden, werden früh und günstig entdeckt, statt an dem Punkt, an dem bereits zu viel ausgegeben wurde, um den Kurs zu ändern.
- Jedes Tor produziert einen frischen, verteidigbaren Business Case, sodass Unterstützung nicht dauerhaft auf der Glaubwürdigkeit des ursprünglichen ruhen muss.

**Kosten und Risiken:**

- Torvorbereitung und -überprüfung verbrauchen echten Aufwand und Kalenderzeit, und der Overhead ist unverhältnismäßig, wenn die Tranchen zu klein sind.
- Gestufte Finanzierung kann bei jedem Tor aus Gründen entzogen werden, die nichts mit der Arbeit zu tun haben — eine Budgetrunde, ein Sponsorwechsel —, was das System halb migriert zurücklässt, was schlechter ist als beide Endzustände.
- Die Sequenzierung, sodass jede Stufe für sich steht, ist echt schwieriger als ein direkter Plan, und manchmal ist es nicht möglich.
- Organisationen, die jedes Tor als Genehmigungsformalität behandeln, bekommen den Overhead ohne die Entscheidung, was der häufigste Fehlermodus ist.
- Der gestufte Ansatz kann insgesamt langsamer und teurer sein als ein verpflichtetes Programm, wenn das Programm erfolgreich gewesen wäre.

## How It Could Be

Die Auftragsverwaltungsplattform eines Einzelhändlers war Gegenstand von drei Modernisierungsvorschlägen über fünf Jahre gewesen, jeder bat um zwischen 4 und 7 Millionen Euro, jeder wurde abgelehnt mit der Begründung, dass die Schätzung nicht glaubwürdig sei. Der vierte Vorschlag bat um 280.000 Euro und vier Monate, um eine Fähigkeit — Produktverfügbarkeitsabfrage — hinter einer Schnittstelle zu extrahieren, sie parallel zur bestehenden Implementierung laufen zu lassen und drei Dinge am Tor zu berichten: was es tatsächlich kostete, was der Vergleich über verstecktes Verhalten offenbarte, und eine überarbeitete Schätzung für den Rest. Die Tranche fand elf undokumentierte Konsumenten der Verfügbarkeitslogik und kostete 40 Prozent mehr als geplant. Die überarbeitete Gesamtprogramm-Schätzung kam auf 9,3 Millionen Euro heraus, wesentlich höher als jeder vorherige Vorschlag. Sie wurde genehmigt, weil die Zahl zum ersten Mal aus der Durchführung eines Teils der Arbeit abgeleitet wurde statt aus Analyse.

Der Tor-Mechanismus bewies sich zwei Tranchen später. Eine Stufe, die die Preis-Engine migrieren sollte, berichtete an ihrem Tor, dass der Ansatz nicht funktionierte: Die Preisregeln waren mit der Promotions-Engine auf Weisen verflochten, die eine unabhängige Extraktion unpraktikabel machten, und die Tranche hatte ihr Budget für die Feststellung dessen verbraucht. Die Empfehlung war, diese Linie zu stoppen und neu zu sequenzieren, zuerst Promotions zu nehmen. Der Sponsor akzeptierte es. Zwei Dinge folgten. Der neu sequenzierte Ansatz funktionierte, und die Tatsache, dass ein Tor genutzt worden war, um etwas zu stoppen, ohne dass jemandem die Schuld gegeben wurde, bedeutete, dass nachfolgende Torberichte merklich offener waren — eine spätere Tranche meldete eine Kostenüberschreitung an ihrem Mittelpunkt statt an ihrem Tor, was eine Korrektur erlaubte, die sonst ein Quartal zu spät angekommen wäre.
