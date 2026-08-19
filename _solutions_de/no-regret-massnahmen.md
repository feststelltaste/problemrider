---
title: No-Regret-Maßnahmen
description: Identifikation der Modernisierungsschritte, die sich unter jeder
  plausiblen Zukunft auszahlen, und diese zuerst umsetzen, solange das Ziel noch
  unentschieden ist.
category:
- Architecture
- Management
- Process
problems:
- modernization-roi-justification-failure
- modernization-strategy-paralysis
- difficulty-quantifying-benefits
- analysis-paralysis
- decision-paralysis
- system-stagnation
- delayed-decision-making
- inability-to-innovate
- accumulated-decision-debt
- second-system-effect
- increasing-brittleness
- technology-lock-in
- legacy-system-documentation-archaeology
- market-pressure
- technology-stack-fragmentation
- upgrade-blocked-by-customization
layout: solution
lang: de
en_slug: no-regret-moves
related_solutions:
- slug: modernization-options-comparison
  similarity: 0.75
- slug: strangler-fig-pattern
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
- slug: boring-technologies
  similarity: 0.7
- slug: staged-investment-with-decision-gates
  similarity: 0.7
- slug: architecture-roadmap
  similarity: 0.65
---

## Description

Eine No-Regret-Maßnahme ist ein Arbeitsstück, das sich zu tun lohnt, unabhängig davon, für welche strategische Option sich die Organisation schließlich entscheidet. Sie existiert als benannte Praxis, weil Legacy-Modernisierungsentscheidungen routinemäßig an einer Frage blockiert werden, die noch nicht beantwortet werden kann — ersetzen oder neu schreiben, Cloud oder On-Premises, kaufen oder bauen —, und solange diese Frage offen ist, geschieht nichts. Dies ist eine falsche Einschränkung. Ein erheblicher Teil dessen, was jede dieser Zukünfte erfordert, ist allen gemeinsam: zu wissen, was das System tut, Tests darum zu haben, die verflochtenen Teile zu trennen, zu entfernen, was nichts nutzt. Diese Arbeit kann sofort beginnen, braucht keine strategische Entscheidung und verbessert die Position der Organisation unter jedem Zweig. Sie macht üblicherweise auch die blockierte Entscheidung leichter, weil der Grund, warum sie nicht beantwortet werden kann, typischerweise ist, dass niemand das System gut genug versteht.

*Die Rahmung von No-Regret-Maßnahmen als eigenständige Entscheidungsklasse stammt aus der Strategie unter Unsicherheit und erscheint als Muster in der Cloud-Native-Transformationsgemeinschaft.*

## How to Apply ◆

> Die lähmende Frage in der Legacy-Modernisierung betrifft fast immer das Ziel, und ein überraschender Anteil der Reise ist identisch, welches Ziel auch gewählt wird.

- **Zählen Sie die strategischen Optionen ehrlich auf**, einschließlich nichts zu tun. Drei oder vier reichen. Der Punkt ist nicht, jetzt zwischen ihnen zu wählen, sondern etwas Konkretes zu haben, gegen das Kandidatenarbeit getestet werden kann.
- **Testen Sie jeden Kandidaten gegen jede Option**: Wäre das noch die Mühe wert gewesen, wenn wir diesen Weg gehen? Arbeit, die alle überlebt, ist eine No-Regret-Maßnahme. Arbeit, die die meisten überlebt, ist eine Low-Regret-Maßnahme und gehört in eine zweite Stufe.
- **Schauen Sie zuerst auf die wiederkehrenden Kategorien.** Characterization Tests rund um das zu bewahrende Verhalten, Entfernung von Code und Features, die nichts nutzt, das Brechen von Abhängigkeiten, die alles schwer verschiebbar machen, die Dokumentation dessen, was das System tatsächlich tut, und die Etablierung von Messung — diese werden für Ersatz, Neuschreibung, Kapselung und fortgesetzte Wartung gleichermaßen gebraucht.
- **Beziehen Sie die Messarbeit explizit ein.** Basislinien, Kostendaten und Nutzungsinstrumentierung sind reine No-Regret-Arbeit: Sie sind günstig, sie werden gebraucht, um irgendetwas zu rechtfertigen, und sie werden unter jeder Option gebraucht, einschließlich nichts zu tun.
- **Beginnen Sie sofort und getrennt von der strategischen Entscheidung.** Der gesamte Wert ist, dass diese Arbeit nicht wartet. Sie an ein Programm zu binden, das Genehmigung erfordert, führt die Blockade wieder ein, die sie umgehen sollte.
- **Berichten Sie sie als Fortschritt gegenüber der strategischen Frage**, nicht als unzusammenhängende Wartung. Sechs Monate No-Regret-Arbeit verändern materiell, was die schließliche Entscheidung kostet, und diese Rahmung hält sie finanziert und die Entscheidung lebendig.
- **Speisen Sie, was Sie lernen, zurück in die Optionen.** Der zuverlässigste Effekt dieser Arbeit ist, dass die strategische Frage beantwortbar wird — meist weil sich herausstellt, dass das System anders ist, als alle angenommen hatten. Testen Sie die Optionen erneut, während Evidenz eintrifft.
- **Achten Sie auf Pseudo-No-Regret-Arbeit.** Alles, was Festlegung auf eine spezifische Zieltechnologie, ein Framework oder einen Anbieter erfordert, ist keine No-Regret-Maßnahme, egal wie es gerahmt wird, und dies ist der häufigste Weg, wie das Konzept missbraucht wird, um eine bevorzugte Richtung einzuschmuggeln.
- **Setzen Sie eine Grenze.** No-Regret-Arbeit kann die strategische Entscheidung nicht unbegrenzt ersetzen. Vereinbaren Sie im Voraus ungefähr, wie lange sie läuft, bevor die Entscheidung erzwungen werden muss, sonst wird sie zu einem bequemen Weg, nie zu entscheiden.

## Tradeoffs ⇄

> Zu tun, was sich unter jeder Zukunft auszahlt, durchbricht Modernisierungsparalyse und braucht keine Genehmigung eines Ziels, kann aber auch zu einem Weg werden, die Entscheidung für immer aufzuschieben.

**Vorteile:**

- Fortschritt beginnt ohne die strategische Entscheidung, was häufig der Unterschied ist zwischen einem System, das verbessert wird, und einem, das weitere zwei Jahre diskutiert wird.
- Jede Option wird günstiger, sodass die Arbeit nicht verschwendet ist, egal wie die schließliche Entscheidung ausfällt.
- Die Entscheidung selbst wird meist beantwortbar, weil die blockierende Unsicherheit normalerweise Unwissenheit über das System ist statt echter strategischer Mehrdeutigkeit.
- Die Arbeit ist einzeln leicht zu rechtfertigen, da jedes Stück für sich verteidigbar ist, ohne Bezug auf ein umstrittenes Programm.
- Optionalität bleibt erhalten. Das Entwirren von Abhängigkeiten und das Entfernen toten Codes erweitert die Menge verbleibender verfügbarer Zukünfte.

**Kosten und Risiken:**

- Sie kann zu einem Ersatz fürs Entscheiden werden, was eine Organisation jahrelang produktiv fühlen lässt, während die strategische Frage offen bleibt.
- Echte No-Regret-Arbeit ist eine kleinere Kategorie, als sie zunächst erscheint, und die Grenze lässt sich leicht zugunsten dessen verwischen, was jemand ohnehin schon tun wollte.
- Manches stellt sich trotzdem als verschwendet heraus — Tests um ein Modul, das schließlich gelöscht wird, Dokumentation eines Systems, das vollständig ersetzt wird.
- Sie produziert über einen längeren Zeitraum kein sichtbares Geschäftsergebnis, was sie in jeder Finanzierungsprüfung verwundbar macht, die fragt, was geliefert wurde.
- Zuerst die einfache gemeinsame Arbeit zu tun kann die schwierigsten, optionsspezifischsten Probleme vollständig unangetastet lassen, sodass die verbleibende Entscheidung nicht weniger einschüchternd ist als zuvor.

## How It Could Be

Ein Versicherer verbrachte zwei Jahre unfähig zu entscheiden, ob er sein Policenverwaltungssystem durch ein Paket ersetzen, es neu schreiben oder es kapseln und behalten sollte. Drei Beratungsunternehmen hatten drei Empfehlungen produziert. Währenddessen änderte sich nichts. Ein neuer Architekt schlug vor, die Frage sechs Monate beiseitezulegen und Kandidatenarbeit gegen alle drei Optionen zu testen. Vier Kategorien überlebten: Characterization Tests rund um die Prämienberechnung, Löschung von Produktvarianten, die Nutzungsdaten zeigten, seit 2015 nicht mehr verkauft worden waren, Trennung der Policendaten von den Reporting-Extrakten, die sie direkt lasen, und Instrumentierung des Systems, um Wartungskosten- und Vorfallbasislinien zu etablieren. Nichts davon brauchte die strategische Entscheidung. Alles davon wurde unter jeder Option gebraucht.

Die Entscheidung beantwortete sich im fünften Monat selbst. Die Bereinigung toter Produkte entfernte 40 Prozent der Verzweigung der Prämienberechnung, und die Characterization-Arbeit etablierte, dass die verbleibenden Regeln weit standardmäßiger waren, als irgendjemand geglaubt hatte — die Komplexität, die alle als Grund angeführt hatten, warum ein Paket nicht passen könnte, waren größtenteils Varianten, die niemand verkaufte. Die Paketoption, zuvor als untauglich verworfen, wurde zur Empfehlung, und der Geschäftsfall wurde auf der während dieser fünf Monate geleisteten Messarbeit aufgebaut. Die spätere Einschätzung des Architekten war, dass die zwei Jahre Stillstand nicht durch eine harte strategische Wahl verursacht worden waren, sondern dadurch, dass niemand genug über das System wusste, um sie zu treffen, und dass dies der Normalfall ist.
