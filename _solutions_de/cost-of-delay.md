---
title: Cost of Delay
description: Quantifizierung, was jeder Monat des Nicht-Handelns kostet, sodass Aufschub
  zu einer bepreisten Entscheidung wird statt zu einer kostenlosen.
category:
- Business
- Management
- Process
problems:
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- short-term-focus
- system-stagnation
- delayed-value-delivery
- competing-priorities
- increased-time-to-market
- high-maintenance-costs
- increasing-brittleness
- inability-to-innovate
- delayed-decision-making
- competitive-disadvantage
- maintenance-cost-increase
- budget-overruns
- increased-cost-of-development
- invisible-nature-of-technical-debt
- legacy-skill-shortage
- market-pressure
- obsolete-technologies
- project-resource-constraints
- regulatory-compliance-drift
- resource-waste
- single-points-of-failure
- slow-development-velocity
- technology-lock-in
- vendor-dependency
- vendor-dependency-entrapment
- wasted-development-effort
- high-technical-debt
- core-modification-of-standard-software
- upgrade-blocked-by-customization
layout: solution
lang: de
en_slug: cost-of-delay
related_solutions:
- slug: total-cost-of-ownership-transparency
  similarity: 0.75
- slug: risk-quantification
  similarity: 0.7
- slug: technical-debt-backlog
  similarity: 0.7
- slug: explicit-prioritization-framework
  similarity: 0.7
- slug: modernization-options-comparison
  similarity: 0.7
- slug: debt-remediation-estimation
  similarity: 0.65
---

## Description

Cost of Delay ist das Geld, das eine Entscheidung pro Zeiteinheit kostet, in der sie nicht getroffen wird — pro Monat des Aufschubs, pro Quartal des Wartens. Es formuliert die Frage neu, die Modernisierungsvorschläge immer verlieren. Gefragt „was ist der Return on Investment", gibt eine Legacy-Verbesserung eine schwache Antwort, weil ihr Nutzen diffus ist und langsam eintrifft. Gefragt „was kostet es uns, weitere sechs Monate zu warten", gibt dieselbe Arbeit eine starke Antwort, weil die Kosten des Wartens konkret sind und bereits bezahlt werden. Die Asymmetrie ist wichtig, weil Aufschub normalerweise als die kostenlose Option behandelt wird: Eine Entscheidung zu verschieben scheint nichts zu kosten, also wird sie standardmäßig gewählt, wiederholt, bis die Situation sich selbst erzwingt. Dem Warten eine monatliche Zahl anzuhängen entfernt diese Illusion und stellt Aufschub auf dieselbe Grundlage wie jede andere Ausgabenentscheidung.

## How to Apply ◆

> Der Cost of Delay eines Legacy-Systems wird meist bereits in Wartungsstunden, Vorfallzeit und Workarounds bezahlt — die Arbeit ist Arithmetik auf Zahlen, die die Organisation hat, keine Prognose.

- **Bauen Sie die Zahl aus bereits angefallenen Kosten auf**, nicht aus prognostiziertem Nutzen. Wartungsaufwand für das zu ersetzende System, ihm zurechenbare Vorfallstunden, Lizenzen für das, was stillgelegt würde, und manueller Workaround-Aufwand im Geschäftsbetrieb sind alle heute messbar und stoppen alle, wenn die Arbeit erledigt ist.
- **Fügen Sie die wachsenden Kosten hinzu.** Manche Komponenten des Aufschubs nehmen über die Zeit zu: ein End-of-Support-Datum, nach dem Patchen maßgeschneidert wird, ein Fachkräftepool, der jedes Jahr schrumpft, ein Datenvolumen, das ein System an eine harte Grenze drängt. Ein steigender Cost of Delay ist ein weit stärkeres Argument als ein flacher, und Legacy-Kosten steigen fast immer.
- **Beziehen Sie die deadline-getriebenen Komponenten explizit ein.** Regulatorische Fristen, Vertragsabläufe und Herstellersupport-Enddaten verwandeln eine graduelle Kostenentwicklung in eine Klippe. Modellieren Sie diese als separaten Term: Die Kosten sind bescheiden bis zu einem Datum, dann sehr groß. Entscheidungsträger reagieren anders auf Klippen als auf Steigungen, und das zu Recht.
- **Drücken Sie es pro Monat aus.** „Dies kostet uns etwa 40.000 € im Monat, es nicht zu tun" ist ein Satz, mit dem eine Finanzfunktion arbeiten kann. Eine annualisierte oder Gesamtzahl lädt zur Debatte über den Zeithorizont ein; eine monatliche Rate tut das nicht.
- **Seien Sie konservativ und zeigen Sie die Komponenten.** Eine aus vier separat überprüfbaren Zahlen aufgebaute Zahl übersteht Prüfung; eine große, selbstbewusste Zahl nicht. Wo eine Komponente eine Schätzung ist, sagen Sie das und nutzen Sie das untere Ende — das Argument braucht selten die aggressive Version.
- **Nutzen Sie es zur Sequenzierung, nicht nur zur Rechtfertigung.** Wenn mehrere Arbeiten jeweils einen Cost of Delay haben, maximiert das Erledigen in absteigender Reihenfolge von Cost-of-Delay-pro-Aufwand, was die Organisation zu zahlen vermeidet. Dies verwandelt die Technik von einem einmaligen Argument in einen Priorisierungsinput.
- **Trennen Sie den Cost of Delay von den Kosten der Arbeit.** Es sind unterschiedliche Zahlen, die unterschiedliche Fragen beantworten, und sie zu vermengen erzeugt einen Business Case, der leicht anzugreifen ist. Der Vergleich, den der Entscheidungsträger braucht, ist zwischen den beiden.
- **Formulieren Sie es periodisch für aufgeschobene Punkte neu.** Ein elf Monate aufgeschobener Punkt, mit den angehäuften Kosten dieses Aufschubs angegeben, macht eine Entscheidung sichtbar, die die Organisation implizit getroffen hat. Dies ist häufig das, was ihn schließlich bewegt.
- **Erzeugen Sie keine künstliche Dringlichkeit.** Ein Cost of Delay, der größer präsentiert wird, als er ist, wird entdeckt, und die verlorene Glaubwürdigkeit gilt für jede nachfolgende Zahl, die das Team produziert.

## Tradeoffs ⇄

> Aufschub zu bepreisen ist das stärkste verfügbare Argument für Legacy-Arbeit, aber es hängt von Messungen ab, die die Organisation möglicherweise nicht hat, und kann zu Fürsprache aufgeblasen werden.

**Vorteile:**

- Es formuliert das Argument von unsicherem zukünftigem Nutzen zu bereits bezahlten Kosten neu, was die Rahmung ist, in der Legacy-Arbeit tatsächlich gewinnen kann.
- Aufschub hört auf, kostenlos zu sein. Die Standardoption erhält einen Preis, was ändert, wie sich Priorisierungsdiskussionen auflösen.
- Steigende und klippenförmige Kosten machen Timing explizit und verwandeln „irgendwann" in eine datierte Entscheidung.
- Es liefert eine verteidigbare Sequenzierungsregel, wenn mehrere Verbesserungen konkurrieren, basierend darauf, was jede vermeidet, statt auf Fürsprache.
- Die Komponenten sind einzeln überprüfbar, was die Glaubwürdigkeit aufbaut, die sich auf den nächsten Vorschlag überträgt.

**Kosten und Risiken:**

- Es erfordert Kostendaten — Wartungsaufwand, Vorfallstunden, Workaround-Zeit —, die viele Organisationen nicht erfassen, sodass die Messung zuerst kommen muss.
- Manche Komponenten sind echte Schätzungen, und ein unwohlwollender Leser wird die schwächste angreifen und die gesamte Zahl verwerfen.
- Die Technik lädt zur Aufblähung ein, und ein später entdeckter überzogener Cost of Delay schadet der Glaubwürdigkeit mehr, als nie einen produziert zu haben.
- Nicht alles Wertvolle hat einen quantifizierbaren Cost of Delay, und eine Kultur, die einen für jeden Vorschlag verlangt, wird systematisch Arbeit aushungern, deren Wert real, aber nicht bepreisbar ist.
- Klippenförmige Kosten können genutzt werden, um falsche Dringlichkeit um Daten herum zu erzeugen, die tatsächlich verhandelbar sind.

## How It Could Be

Ein Team hatte drei Jahre in Folge den Ersatz eines Batch-Scheduling-Systems vorgeschlagen und wurde jedes Mal aus demselben Grund abgelehnt: Der Return sei unklar. Beim vierten Versuch hörten sie auf, über Return zu argumentieren, und bepreisten den Aufschub. Vier Komponenten, jede aus Daten, die sie bereits hatten: 1,8 Entwicklertage im Monat für die Wartung der maßgeschneiderten Retry-Logik des Schedulers; 34 Vorfallstunden pro Quartal, die ihm zurechenbar waren, etwa die Hälfte davon außerhalb der Arbeitszeit; ein Supportvertrag für eine Version, die der Hersteller bereits als End-of-Life erklärt hatte, jährlich zu einer Prämie verlängert; und — die Zahl, die niemand zuvor in ein Dokument geschrieben hatte — etwa 1,2 Tage im Monat, in denen ein Betriebsanalyst manuell Jobs abglich, die still fehlgeschlagen waren. Die Summe belief sich auf etwa 31.000 € im Monat, steigend, mit einem Stufenanstieg beim Support-Ende vierzehn Monate später. Der Vorschlag wurde im selben Meeting genehmigt, in dem er präsentiert wurde.

Der Sequenzierungsnutzen erwies sich im folgenden Jahr als wichtiger. Das Team berechnete den Cost of Delay für sechs aufgeschobene Verbesserungen und stellte fest, dass die, für die sie am stärksten geworben hatten — ein Testinfrastruktur-Umbau — an vierter Stelle rangierte, während eine kleine, unglamouröse Korrektur an einem Datenexport, den drei Fachabteilungen manuell korrigierten, mit weitem Abstand an erster Stelle rangierte. Diese Korrektur dauerte neun Tage und eliminierte etwa zwei volle Tage Arbeit pro Woche in Abteilungen, mit denen das Entwicklungsteam nie gesprochen hatte. Sie wäre unter dem vorherigen Ansatz gar nicht vorgeschlagen worden, weil niemand im Engineering den Schmerz erlebte und sie nichts produzierte, was ein Entwickler als interessant beschrieben hätte.
