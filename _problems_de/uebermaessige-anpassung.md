---
title: Übermäßige Anpassung
description: So viel kunden- oder standortspezifisches Verhalten häuft sich an,
  dass keine zwei Installationen gleich sind und jede Änderung gegen jede Variante
  geprüft werden muss.
category:
- Architecture
- Business
- Process
related_problems:
- slug: customization-outside-version-control
  similarity: 0.7
- slug: reimplemented-standard-functionality
  similarity: 0.7
- slug: core-modification-of-standard-software
  similarity: 0.7
- slug: custom-report-sprawl
  similarity: 0.7
- slug: low-code-customization-sprawl
  similarity: 0.65
- slug: upgrade-blocked-by-customization
  similarity: 0.65
solutions:
- explicit-extension-points
- customization-cost-attribution
- variant-consolidation
- feature-usage-measurement
- attribute-usage-analysis
- product-strategy-alignment
- explicit-prioritization-framework
- definition-of-ready
- modularization-and-bounded-contexts
- feature-toggles
- standard-software
- decision-rights-and-escalation
- total-cost-of-ownership-transparency
- large-scale-refactoring
- fit-to-standard-principle
- role-model-rationalization
layout: problem
lang: de
en_slug: excessive-customization
---

## Description

Übermäßige Anpassung entsteht, wenn sich ein System so viel kunden-, standort- oder abteilungsspezifisches Verhalten anhäuft, dass es kein einziges Produkt mehr gibt – es gibt eine Familie divergierender Varianten, die zufällig denselben Namen teilen. Jede einzelne Anpassung war begründet: Ein Kunde brauchte etwas, ein Deal hing davon ab, eine Abteilung hatte einen wirklich anderen Prozess. Zusammengenommen zerstören sie die Ökonomie des Produkts, weil jede Änderung nun gegen jede Variante entworfen, gegen jede Variante getestet und auf Installationen deployt werden muss, die sich jeweils leicht unterschiedlich verhalten. Der Zustand ist selbstverstärkend. Sobald das Upgrade teuer ist, fallen Kunden zurück, und jede Installation driftet weiter von jeder anderen weg, was das nächste Upgrade noch teurer macht.

## Indicators ⟡

- Keine zwei Installationen laufen mit derselben Konfiguration, und niemand kann eine Liste der Unterschiede erstellen
- Die Schätzung einer Änderung erfordert die Frage, welche Kunden sie betrifft, und die Antwort braucht Tage
- Manche Kunden sind mehrere Versionen zurück, und ihr Upgrade wird als Projekt statt als Routine behandelt
- Vertriebszusagen beinhalten regelmäßig Verhalten, das noch nicht existiert und nur für einen Kunden gelten wird
- Die Testsuite hat kundenspezifische Fälle, oder schlimmer, das Testen erfolgt pro Installation nach dem Deployment
- Neuen Entwicklern wird gesagt, dass ein Modul "für den Großkunden anders funktioniert", ohne Dokument, das erklärt, wie
- Niemand kann sagen, was das Standardprodukt tut, ohne es zu qualifizieren

## Symptoms ▲

- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Jede Variante trägt ihre eigene Wartungslast, und die Gesamtsumme wächst mit der Anzahl der Installationen statt mit der Größe des Produkts.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Eine Änderung, die in einem Ein-Varianten-Produkt klein wäre, muss gegen jede abweichende Installation entworfen, implementiert und verifiziert werden.
- [Testkomplexität](testkomplexitaet.md)
<br/>  Die Anzahl der zu verifizierenden Konfigurationen vervielfacht sich mit jeder Anpassung, und volle Abdeckung der Kombinationen wird schnell unmöglich.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Neue Funktionalität muss jede bestehende Variante berücksichtigen, bevor sie ausgeliefert werden kann, was unkomplizierte Arbeit zu einer Übung in Kompatibilität macht.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Defekte treten in bestimmten Varianten unter bestimmten Konfigurationen auf, und die nie getesteten Kombinationen sind genau dort, wo sie sich zeigen.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Eine gegen das Standardprodukt verifizierte Änderung bricht einen Kunden, dessen Variante auf dem vorherigen Verhalten beruhte, auf eine Weise, die niemand aufgezeichnet hatte.
- [Lange Release-Zyklen](lange-release-zyklen.md)
<br/>  Ein Release bedeutet Validierung gegen viele Installationen, was den Zyklus verlängert, bis Releases selten genug werden, um selbst riskant zu sein.
- [Wissenssilos](wissenssilos.md)
<br/>  Jede Variante wird tendenziell nur von demjenigen verstanden, der sie gebaut hat, und dieses Wissen wird selten aufgeschrieben, weil die Variante temporär sein sollte.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Bedingte Verzweigungen für bestimmte Kunden häufen sich in der gesamten Codebasis an und werden nie entfernt, weil niemand sicher ist, wer noch davon abhängt.

## Causes ▼

- [Übermäßige Gefälligkeit gegenüber Stakeholdern](uebermaessige-gefaelligkeit-gegenueber-stakeholdern.md)
<br/>  Jede Kundenanfrage wird akzeptiert, weil eine Ablehnung sich wie schlechter Service anfühlt, und die kumulativen Kosten der Akzeptanz sind im Moment jeder Entscheidung unsichtbar.
- [Marktdruck](marktdruck.md)
<br/>  Wettbewerbsdeals werden gewonnen, indem versprochen wird, alles zu berücksichtigen, was der Interessent verlangt, und die technische Konsequenz kommt erst nach Vertragsunterzeichnung.
- [Feature-Creep](feature-creep.md)
<br/>  Der Umfang weitet sich kontinuierlich aus, ohne dass etwas entfernt wird, und kundenspezifisches Verhalten ist die Form, die diese Ausweitung in einem Produkt mit vielen Installationen annimmt.
- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Anfragen werden wie angegeben umgesetzt statt untersucht, sodass ein Bedürfnis, das mehrere Kunden teilen, mehrfach als mehrere Varianten gebaut wird.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Der unmittelbare Deal ist mehr wert als die langfristige Wartbarkeit des Produkts, und dieser Tausch wird wiederholt von Personen gemacht, die nie dessen Kosten tragen.
- [Chaos in der Produktrichtung](chaos-in-der-produktrichtung.md)
<br/>  Ohne klare Definition dessen, was das Standardprodukt ist, gibt es keine Grundlage, auf der irgendeine Anfrage abgelehnt werden könnte.
- [Vakuum an Projektautorität](vakuum-an-projektautoritaet.md)
<br/>  Niemand hat die Stellung, eine Anpassung abzulehnen, sodass die Standardantwort Ja ist und die Entscheidung nie tatsächlich von jemandem getroffen wird.

## Detection Methods ○

- Zählung der Konfigurations-Flags, Feature-Toggles und kundenspezifischen Verzweigungen in der Codebasis und Prüfung des Trends über die letzten zwei Jahre
- Suche im Code nach Bedingungen, die einen bestimmten Kunden, Standort oder Mandanten benennen – diese sind selten anderswo dokumentiert
- Vergleich der deployten Konfiguration über Installationen hinweg und Zählung der abweichenden Felder
- Messung der Verteilung der Versionsnummern über Installationen; eine breite Streuung deutet darauf hin, dass Upgrades teuer geworden sind
- Nachverfolgung, wie viel des Aufwands jedes Releases in die Berücksichtigung von Varianten fließt statt in neue Fähigkeiten
- Die Frage, wie lange es dauert, "welche Kunden betrifft diese Änderung" zu beantworten – wenn es mehr als eine Stunde ist, wird die Varianz nicht mehr nachverfolgt
- Überprüfung der letzten zehn Kundenzusagen und Zählung, wie viele Verhalten einführten, das genau auf eine Installation zutrifft

## Examples

Ein mittelgroßer Anbieter klinischer Terminplanungssoftware hatte 34 Krankenhausinstallationen, die über elf Jahre aus einer Codebasis gebaut wurden. Jedes Krankenhaus hatte während der Beschaffung Anpassungen ausgehandelt: unterschiedliche Regeln dafür, wie ein stornierter Termin seinen Slot freigab, unterschiedliche Eskalationspfade, unterschiedliche Berichtslayouts, und in vier Fällen eine unterschiedliche Definition dessen, was als abgeschlossener Besuch zählte. Nichts davon war Konfiguration – es war bedingte Logik in der Codebasis, verknüpft mit einer Installationskennung. Das Hinzufügen eines unkomplizierten Features bedeutete, das Modul zu lesen, die sieben kundenspezifischen Verzweigungen darin zu finden und über jede nachzudenken. Ihr Release-Zyklus war von sechs Wochen auf neun Monate gewachsen, und elf Krankenhäuser liefen mit über zwei Jahre alten Versionen, weil die Upgrade-Kosten pro Standort auf mehrere Wochen Beratung angewachsen waren.

Der selbstverstärkende Charakter zeigte sich darin, wie sich die Situation entwickelt hatte. Frühe Anpassungen waren klein, und die Ökonomie des Produkts absorbierte sie. Während die Anzahl wuchs, wurde jedes Release teurer zu validieren, sodass Releases seltener wurden. Seltenere Releases bedeuteten, dass Kunden länger auf Anfragen warteten, was sie hartnäckiger machte, dass ihre Anfragen genau erfüllt wurden, was mehr Anpassungen erzeugte. Als der Anbieter das Muster erkannte, ging etwa 40 Prozent der Engineering-Kapazität in den Abgleich von Varianten, und keine einzige Person konnte beschreiben, was das Standardprodukt tat.
