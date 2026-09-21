---
title: Transparenz der Gesamtbetriebskosten
description: Messung und Veröffentlichung dessen, was ein Legacy-System
  tatsächlich kostet, um am Laufen zu bleiben — Wartung, Vorfälle,
  Lizenzen und verlorene Kapazität — sodass Investitionsentscheidungen
  auf Zahlen statt auf Eindrücken beruhen.
category:
- Business
- Management
- Process
problems:
- budget-overruns
- increased-cost-of-development
- maintenance-cost-increase
- high-maintenance-costs
- invisible-nature-of-technical-debt
- short-term-focus
- system-stagnation
- planning-credibility-issues
- delayed-value-delivery
- inability-to-innovate
- obsolete-technologies
- high-technical-debt
- project-resource-constraints
- stakeholder-confidence-loss
- modernization-roi-justification-failure
- difficulty-quantifying-benefits
- excessive-customization
- voided-vendor-support
layout: solution
lang: de
en_slug: total-cost-of-ownership-transparency
related_solutions:
- slug: cost-of-delay
  similarity: 0.75
- slug: business-metrics
  similarity: 0.75
- slug: technical-debt-backlog
  similarity: 0.7
- slug: risk-quantification
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
- slug: feature-usage-measurement
  similarity: 0.7
---

## Description

Transparenz der Gesamtbetriebskosten ist die Praxis, zu messen, was ein System tatsächlich kostet, um es am Laufen zu halten, und diese Zahl zusammen mit den Kosten seiner Änderung zu veröffentlichen. Die Komponenten sind einzeln banal und zusammen unsichtbar: Entwicklerzeit, die für Wartung statt neue Fähigkeiten aufgewendet wird, Vorfallreaktion und Bereitschaftslast, Lizenzen und Infrastruktur, der durch manuelle Workarounds verbrauchte Aufwand, und die Opportunitätskosten der Kapazität, die dafür draufgeht, das System am Leben zu halten. Organisationen finanzieren routinemäßig die Modernisierung von Systemen, deren Betriebskosten sie nicht beziffern können, und lehnen sie routinemäßig aus demselben Grund ab. Das Argument für Investition in ein Legacy-System scheitert fast immer nicht, weil der Fall schwach ist, sondern weil er qualitativ vorgebracht wird — "der Code ist schlecht, es wird schwieriger" — gegen einen Vorschlag, der mit einer angehängten Zahl ankommt.

## How to Apply ◆

> Die Kosten, die ein Legacy-System teuer machen, sind über Budgets verteilt, die niemand zusammenzählt: Entwicklungsgehälter, Vorfallstunden, Lizenzerneuerungen und die manuelle Arbeit, die Fachabteilungen vor Jahren absorbierten und nicht mehr erwähnen.

- **Teilen Sie den Entwicklungsaufwand nach Kategorie auf** und verfolgen Sie ihn: neue Fähigkeiten, Wartungs- und Defektarbeit, ungeplante Vorfallreaktion und verpflichtende Arbeit wie Compliance und Abhängigkeits-Upgrades. Zwei oder drei konsistent verfolgte Kategorien sind mehr wert als eine detaillierte Taxonomie, die einen Monat lang verfolgt wird. Das Verhältnis ist üblicherweise der Schlagzeilenbefund.
- **Quantifizieren Sie die Vorfalllast** in Stunden, nicht Vorfallanzahlen: Erkennungszeit, Lösungszeit, beteiligte Personen und der Anteil außerhalb der Arbeitszeit. Das Zählen von Vorfällen unterschätzt die Kosten, weil die teuren die langen sind.
- Beziehen Sie die **direkten Betriebskosten** ein, die üblicherweise in einem anderen Budget gehalten werden: Lizenzen, Support-Verträge, Infrastruktur und die Spezialisten-Auftragnehmer, die gehalten werden, weil niemand intern eine Komponente pflegen kann.
- **Finden Sie die manuellen Workarounds außerhalb des Technologiebudgets.** Legacy-Systeme drücken Kosten in die Organisationen, die sie nutzen — der Abgleich, den jemand jeden Monat macht, die Tabellenkalkulation, die existiert, weil ein Bericht es nicht tut. Diese Kosten sind häufig größer als die gesamten IT-Kosten des Systems, und sie sind unsichtbar, bis jemand die Abteilungen fragt.
- Drücken Sie die **Opportunitätskosten explizit aus**: Wenn siebzig Prozent der Kapazität dafür draufgeht, das System am Laufen zu halten, dann sind dreißig Prozent das, was der Organisation für alles zur Verfügung steht, was sie will. Es so auszudrücken verwandelt eine technische Beschwerde in eine Geschäftsbeschränkung.
- **Verfolgen Sie den Trend, nicht nur das Niveau.** Ein Wartungsanteil, der über drei Jahre von vierzig auf sechzig Prozent steigt, ist ein weit überzeugenderes Argument als sechzig Prozent isoliert, weil es vorhersagt, wo die Linie hundert erreicht.
- **Ordnen Sie Kosten Teilen des Systems zu**, wo die Daten es erlauben, unter Nutzung von Änderungshäufigkeit und Vorfalldaten. "Dieses Subsystem ist acht Prozent des Codes und vierzig Prozent der Vorfallstunden" lenkt Investition auf eine Weise, die eine systemweite Zahl nicht kann.
- **Veröffentlichen Sie in fester Taktung** an die Personen, die Finanzierungsentscheidungen treffen, in ihren Begriffen — Kosten, Risiko, Kapazität — statt in technischen Begriffen. Eine Zahl, die einmal in einem Business Case erscheint, ist ein Argument; eine Zahl, die vierteljährlich erscheint, ist ein Managementinstrument.
- **Berichten Sie die Wirkung von Verbesserungen gegen dieselben Maße.** Investition, die keine Bewegung in den Zahlen demonstrieren kann, die sie rechtfertigten, wird kein zweites Mal gewährt werden.

## Tradeoffs ⇄

> Kosten sichtbar zu machen ist üblicherweise der entscheidende Schritt, um Modernisierung finanziert zu bekommen, aber die Messung selbst kostet Aufwand, und die resultierenden Zahlen können auf Weisen genutzt werden, die das Team nicht beabsichtigte.

**Vorteile:**

- Investitionsentscheidungen beruhen auf vergleichbaren Zahlen statt auf konkurrierenden Behauptungen, was die Bedingung ist, unter der Wartungsarbeit gegen Feature-Arbeit gewinnen kann.
- Der Trend prognostiziert den zukünftigen Zustand, was das Argument ist, das Entscheidungen von irgendwann zu jetzt verschiebt.
- Kostenzuordnung lenkt Verbesserungsaufwand auf die Teile des Systems, die tatsächlich das Budget verbrauchen, statt auf die Teile, die am unangenehmsten zu bearbeiten sind.
- Die versteckten organisatorischen Kosten manueller Workarounds werden sichtbar, und sie sind häufig die einzelne größte Zahl in der Analyse.
- Verbesserungen werden im Nachhinein verteidigbar, was die nächste Investition leichter zu erhalten macht.

**Kosten und Risiken:**

- Konsistente Aufwandsverfolgung wird von Entwicklern nicht gemocht und verschlechtert sich schnell, wenn die Kategorien zu feinkörnig sind oder die Daten genutzt werden, um Einzelpersonen zu bewerten.
- Die Zahlen können gegen das Team gewendet werden — ein hoher Wartungsanteil als Ineffizienz gelesen statt als Eigenschaft des Systems —, besonders wo der Leser nach einem Grund für Outsourcing sucht.
- Opportunitätskosten und Workaround-Kosten sind Schätzungen, und Schätzungen werden angegriffen, wenn die Schlussfolgerung unwillkommen ist. Konservative Zahlen mit angegebenen Annahmen überstehen Prüfung besser als aggressive.
- Die Erfassung der organisatorischen Kosten erfordert Kooperation von Abteilungen außerhalb der Technologie, die möglicherweise keinen Anreiz haben, ihre Workarounds sichtbar zu machen.
- Die Messung kann zum Selbstzweck werden und Aufwand für Berichterstattung verbrauchen, der besser für die Reduzierung der berichteten Kosten aufgewendet würde.

## How It Could Be

Eine Entwicklungsmanagerin hatte zweimal versagt, Finanzierung für die Modernisierung eines zwanzig Jahre alten Auftragsverarbeitungssystems zu erhalten. Beim dritten Versuch verbrachte sie sechs Wochen mit Messen statt Argumentieren. Über zwei Quartale verfolgte Entwicklungszeit zeigte 71 Prozent, die in Wartung, Defekte und Vorfälle flossen, gegenüber 52 Prozent drei Jahre zuvor, laut rekonstruierten Ticketdaten. Vorfallstunden summierten sich über die beiden Quartale auf 940, davon 310 außerhalb der Arbeitszeit. Lizenzen und ein gehaltener Spezialisten-Auftragnehmer fügten eine Zahl hinzu, die nie zuvor im selben Dokument wie die Gehälter erschienen war. Und eine Umfrage bei zwei Fachabteilungen fand ungefähr 1,5 Vollzeitäquivalente, die für manuellen Abgleich aufgewendet wurden, der ausschließlich existierte, weil zwei Subsysteme keine Daten korrekt austauschten. Die extrapolierte Trendlinie zeigte, dass Wartung innerhalb von vier Jahren die gesamte verfügbare Kapazität verbrauchen würde. Die Finanzierung wurde in einem Meeting genehmigt.

Die Zuordnungsanalyse änderte dann, wofür die Finanzierung ausgegeben wurde. Das Team hatte angenommen, der Aufwand sollte zum ältesten Subsystem gehen, das am unangenehmsten zu bearbeiten war. Die Kreuzung von Vorfallstunden mit Änderungshäufigkeit zeigte etwas anderes: Das Auftragsvalidierungsmodul, vergleichsweise modern und unauffällig in der Bearbeitung, machte 44 Prozent der Vorfallstunden aus, wegen der Art, wie es mit einer Partnerschnittstelle interagierte. Die Umlenkung der ersten Phase dorthin reduzierte Vorfallstunden innerhalb von zwei Quartalen um ungefähr ein Drittel — ein Ergebnis, das, gegen dieselben Maße berichtet, die die Investition gerechtfertigt hatten, die zweite Phase ohne Business Case sicherte.
