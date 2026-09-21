---
title: Qualitäts-Ratsche
description: Verlangen, dass Qualitätsmaße sich niemals gegenüber dem
  heutigen Stand verschlechtern, statt absolute Schwellenwerte zu setzen,
  die eine Legacy-Codebasis nie erreichen kann.
category:
- Code
- Process
- Testing
problems:
- high-technical-debt
- quality-degradation
- increasing-brittleness
- increased-technical-shortcuts
- poor-test-coverage
- accumulation-of-workarounds
- mixed-coding-styles
- undefined-code-style-guidelines
- quality-compromises
- inconsistent-execution
- copy-paste-programming
- refactoring-avoidance
- maintenance-cost-increase
- brittle-codebase
- convenience-driven-development
- code-duplication
- authorization-role-explosion
- low-code-customization-sprawl
layout: solution
lang: de
en_slug: quality-ratchet
related_solutions:
- slug: code-quality-gates
  similarity: 0.8
- slug: code-metrics
  similarity: 0.75
- slug: static-analysis-and-linting
  similarity: 0.75
- slug: code-coverage-analysis
  similarity: 0.7
- slug: delivery-performance-metrics
  similarity: 0.7
- slug: regression-testing
  similarity: 0.7
---

## Description

Eine Qualitäts-Ratsche erzwingt, dass sich ein Maß nicht gegenüber seinem aktuellen Wert verschlechtern darf, statt zu verlangen, dass es einen absoluten Standard erreicht. Die Testabdeckung darf nicht unter das fallen, was sie heute ist; die Anzahl statischer Analysewarnungen darf nicht steigen; die Zahl der Abhängigkeiten jenseits des Support-Endes darf nicht wachsen. Dies löst den spezifischen Grund, warum Qualitäts-Gates an Legacy-Codebasen scheitern. Ein absoluter Schwellenwert — achtzig Prozent Abdeckung, null Warnungen — ist auf einem System mit zwanzig Jahren Geschichte unerreichbar, sodass er entweder nicht übernommen oder übernommen und sofort ausgesetzt wird, oder mit so vielen Ausnahmen übernommen wird, dass er nichts mehr misst. Eine Ratsche ist vom ersten Tag an erreichbar, unabhängig vom Ausgangspunkt, weil sie nur verlangt, dass die heutige Änderung die Dinge nicht schlimmer macht. Über die Zeit verwandelt sie jede Verbesserung in einen neuen Boden, sodass sich Fortschritt ansammelt, statt zu erodieren.

## How to Apply ◆

> Der Grund, warum Legacy-Qualitätsinitiativen scheitern, ist fast nie, dass das Team dem Standard nicht zustimmt; es ist, dass der Standard unerreichbar ist, und unerreichbare Standards werden ignoriert.

- **Setzen Sie den anfänglichen Schwellenwert auf den aktuell gemessenen Wert**, was auch immer er ist, und zeichnen Sie ihn auf. Eine Ratsche, die bei einer ambitionierten Zahl beginnt, ist ein absolutes Gate in anderer Kleidung und wird auf dieselbe Weise scheitern.
- **Rätschen Sie bei den Maßen, die zählen und verlässlich gemessen werden können**: Testabdeckung, Befunde statischer Analyse, Build-Dauer, Aktualität von Abhängigkeiten und die Anzahl der Dateien, die einen Komplexitätsschwellenwert überschreiten. Zwei oder drei Ratschen reichen; ein Dutzend erzeugt Reibung ohne entsprechenden Nutzen.
- **Wenden Sie sie auf geänderten Code an, wo das Maß es erlaubt.** Abdeckung auf modifizierten Zeilen ist eine weit nützlichere Ratsche als Abdeckung über die gesamte Codebasis, da sie die tatsächlich bearbeiteten Bereiche verbessert, statt Tests für ruhenden Code zu fördern.
- **Erzwingen Sie sie in der Pipeline**, nicht durch Konvention. Eine Ratsche, die davon abhängt, dass Menschen sich erinnern, wird still erodieren, und die Erosion ist unsichtbar, bis jemand misst.
- **Aktualisieren Sie den Boden automatisch, wenn eine Änderung das Maß verbessert.** Dies ist der Mechanismus: eine aus eigenen Gründen vorgenommene Verbesserung wird permanent, ohne dass jemand entscheidet, sie zu schützen.
- **Bieten Sie eine explizite Ausnahme mit Namen und angehängtem Grund**, und überprüfen Sie die Ausnahmen periodisch. Eine Ratsche ohne Notausgang wird beim ersten echten Notfall umgangen oder deaktiviert; eine, deren Ausnahmen aufgezeichnet werden, bleibt ehrlich und zeigt, wo die Reibung liegt.
- **Rätschen Sie nicht bei Maßen, die leicht zu manipulieren sind.** Zeilenabdeckung lädt zu Tests ein, die Code ausführen, ohne irgendetwas zu behaupten. Wo ein Maß ohne die zugrunde liegende Verbesserung erfüllt werden kann, wird es das, und die Ratsche erzwingt dann ein Ritual.
- **Führen Sie sie eine nach der anderen ein**, mit einer Berichtsperiode vor der Durchsetzung. Eine Ratsche, die am Tag ihrer Einführung Builds fehlschlagen lässt, wird für alles beschuldigt, was in dieser Woche sonst schiefgeht.
- **Berichten Sie die Bewegung des Bodens** vierteljährlich. Der Trend — Abdeckungsboden, der über ein Jahr ohne dediziertes Testprojekt von 31 auf 44 Prozent steigt — ist der Beweis, dass der Mechanismus funktioniert, und er ist ungewöhnlich überzeugend, weil niemand überzeugt werden musste, ihn zu erzeugen.

## Tradeoffs ⇄

> Ratschen sind auf jeder Codebasis erreichbar und machen Verbesserung permanent, aber sie verhindern nur Verschlechterung — sie produzieren nicht von selbst Fortschritt, und sie fügen Reibung an unpassenden Momenten hinzu.

**Vorteile:**

- Sie ist ab dem ersten Tag auf jeder Codebasis übernehmbar, wie schlecht auch immer, was genau das ist, was absolute Schwellenwerte nicht sind.
- Verbesserungen werden permanent. Ohne eine Ratsche erodieren Gewinne, die während eines fokussierten Aufwands gemacht wurden, im folgenden Jahr, und der Aufwand muss wiederholt werden.
- Verschlechterung hört auf, unsichtbar zu sein. Legacy-Qualität verfällt schrittweise, wobei jede einzelne Änderung verteidigbar ist, und die Ratsche ist das, was das Aggregat sichtbar macht.
- Die Bewegung des Bodens über die Zeit ist ein evidenzbasiertes Fortschrittsmaß, das nichts zusätzlich kostet zu produzieren.
- Sie übt Druck am Änderungspunkt aus, wo der Entwickler den Kontext hat, statt durch periodische Bereinigungskampagnen.

**Kosten und Risiken:**

- Eine Ratsche verhindert Verschlechterung, treibt aber keine Verbesserung an. Ein Team kann unbegrenzt auf seinem anfänglichen Boden sitzen und vollständig konform sein.
- Sie fügt genau dann Reibung hinzu, wenn Menschen in Eile sind, was ist, wenn Ausnahmen genutzt werden und die Praxis am wahrscheinlichsten aufgegeben wird.
- Manipulierbare Maße werden manipuliert, und eine Ratsche auf einem schwachen Proxy erzwingt den Anschein von Qualität statt Qualität.
- Codebasisweite Maße können auf unhilfreiche Weisen erfüllt werden, wie das Hinzufügen von Tests zu trivialem ruhendem Code, um einen Rückgang in einem wichtigen Bereich auszugleichen.
- Legitime Arbeit macht manchmal ein Maß schlechter — das Löschen gut getesteten Codes kann die Gesamtabdeckung senken —, und eine Ratsche ohne Urteilsvermögen wird Verbesserungen blockieren.

## How It Could Be

Ein Team hatte zweimal ein Abdeckungs-Gate versucht. Das erste setzte 70 Prozent, gegen eine tatsächliche Zahl von 23 Prozent, und wurde innerhalb einer Woche aufgegeben. Das zweite setzte 30 Prozent, was durch das Hinzufügen von Tests zu einem von niemandem genutzten Utility-Paket erfüllt wurde, wonach die Abdeckung in den wichtigen Bereichen weiter fiel. Der dritte Versuch war eine Ratsche: Die Abdeckung auf modifizierten Zeilen durfte nicht unter den aktuellen Wert für diese Datei fallen, und die Gesamtabdeckung durfte nicht unter 23 Prozent fallen. Sie ließ im ersten Monat keine Builds fehlschlagen, weil sie von niemandem etwas Neues verlangte — nur, dass Änderungen ihre Datei nicht verschlechtern. Über vierzehn Monate stieg die Gesamtabdeckung auf 44 Prozent, ohne dediziertes Testprojekt, vollständig als Nebenprodukt gewöhnlicher Arbeit in den Bereichen, die bearbeitet wurden.

Die Ratsche für statische Analyse produzierte eine andere Lektion. Die Codebasis hatte ungefähr 4.100 Warnungen, und der vorherige Versuch des Teams, sie zu reduzieren, war nach einer Woche Mühsal ins Stocken geraten. Eine Ratsche auf die Gesamtzahl bedeutete, dass eine Änderung, die eine Warnung hinzufügte, eine entfernen musste, was Entwickler im Allgemeinen erfüllten, indem sie etwas Angrenzendes zu dem behoben, was sie bereits anfassten. Die Zahl fiel über ein Jahr auf etwa 2.600. Das Ausnahmeprotokoll erwies sich als das wertvollere Artefakt: 31 Ausnahmen, von denen 19 dieselbe Warnungsklasse im selben Subsystem waren — ein Codegenerierungsschritt, der Ausgaben produzierte, gegen die der Analyzer protestierte und die niemand ändern konnte. Das wurde durch die Konfiguration eines Ausschlusses für generierten Code behoben, der die Beziehung des Teams zur statischen Analyse jahrelang still vergiftet hatte.
