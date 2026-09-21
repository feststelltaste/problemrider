---
title: Kernmodifikation von Standardsoftware
description: Der Code des Herstellers wurde direkt verändert, statt über unterstützte
  Mechanismen erweitert zu werden, sodass jedes Update mit lokalen Änderungen kollidiert.
category:
- Architecture
- Dependencies
- Code
related_problems:
- slug: upgrade-blocked-by-customization
  similarity: 0.7
- slug: excessive-customization
  similarity: 0.7
- slug: voided-vendor-support
  similarity: 0.65
- slug: reimplemented-standard-functionality
  similarity: 0.65
- slug: customization-outside-version-control
  similarity: 0.6
- slug: process-software-misfit
  similarity: 0.6
solutions:
- explicit-extension-points
- fit-to-standard-principle
- customization-under-version-control
- large-scale-refactoring
- characterization-tests
- change-impact-analysis
- technical-debt-assessment
- debt-remediation-estimation
- customization-cost-attribution
- variant-consolidation
- vendor-management-practice
- modernization-options-comparison
- cost-of-delay
layout: problem
lang: de
en_slug: core-modification-of-standard-software
---

## Description

Kernmodifikation entsteht, wenn ein erworbenes Softwareprodukt angepasst wird, indem der vom Hersteller gelieferte eigene Code bearbeitet wird, statt die vom Hersteller bereitgestellten Erweiterungsmechanismen zu nutzen. Es ist in dem Moment, in dem es gewählt wird, meist der schnellste Weg: Das benötigte Verhalten befindet sich in einer gelieferten Routine, die Änderung dort dauert eine Stunde, und der Bau desselben über einen unterstützten Erweiterungspunkt dauert eine Woche. Die Kosten kommen später und dauerhaft. Jedes nachfolgende Herstellerupdate überschreibt die Modifikation oder kollidiert mit ihr, sodass jedes Upgrade zu einer Übung im Abgleich zweier Änderungssätze am selben Code wird. Weil die Organisation nun einen Fork von Software pflegt, die sie nicht selbst geschrieben hat und nicht vollständig versteht, kann der Fork nie versöhnt werden – er kann nur weitergetragen werden.

## Indicators ⟡

- Das Anwenden eines Herstellerupdates erfordert einen Merge, und jemand muss pro Konflikt entscheiden, welche Version gewinnt
- Es gibt eine formale oder informelle Liste "von uns geänderter Objekte", die vor jedem Upgrade konsultiert werden muss
- Upgrade-Projekte werden über Monate geplant und beziehen externe Berater ein, unabhängig davon, wie klein das Release ist
- Die Herstellerdokumentation beschreibt nicht, wie sich das eigene System verhält, und die Mitarbeitenden haben gelernt, ihr zu misstrauen
- Modifikationen enthalten Kommentare von Entwicklern, die vor Jahren das Unternehmen verlassen haben, mit Erklärungen zu Geschäftsregeln, die nicht mehr gelten
- Die Organisation betreibt eine Version, die weit hinter dem aktuellen Release liegt, ohne einen datierten Plan, die Lücke zu schließen

## Symptoms ▲

- [Veraltete Technologien](veraltete-technologien.md)
<br/>  Weil jedes Upgrade den Fork versöhnen muss, werden Upgrades aufgeschoben, und die installierte Version fällt zunehmend hinter das zurück, was der Hersteller unterstützt.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Die Organisation pflegt Code, den sie nicht geschrieben hat, in einem System, das sie nicht vollständig versteht, zusätzlich zu ihren eigenen Erweiterungen.
- [Testkomplexität](testkomplexitaet.md)
<br/>  Von modifiziertem Herstellercode kann nicht angenommen werden, dass er sich wie dokumentiert verhält, sodass die Verifikation auch Standardverhalten abdecken muss, das der Hersteller bereits getestet hat.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Das Abgleichen eines Herstellerupdates mit lokalen Modifikationen führt Fehler wieder ein, die der Hersteller behoben hat, oder entfernt lokales Verhalten, auf das sich etwas verlassen hat.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Niemand ist sich sicher, wofür eine modifizierte Routine nun verantwortlich ist, sodass Änderungen in ihrer Nähe vermieden werden, selbst wenn sie nötig sind.
- [Lange Release-Zyklen](lange-release-zyklen.md)
<br/>  Der mit jedem Herstellerrelease verbundene Abgleichs- und Regressionsaufwand macht routinemäßige Updates zu Projekten, die nur selten durchgeführt werden können.
- [Gefangenschaft durch Anbieterabhängigkeit](gefangenschaft-durch-anbieterabhaengigkeit.md)
<br/>  Eine stark modifizierte Installation kann nicht durch ein vergleichbares Produkt ersetzt werden, ohne jede Modifikation neu zu machen, was die Option des Wechsels beseitigt.
- [Schwer verständlicher Code](schwer-verstaendlicher-code.md)
<br/>  Herstellercode mit lokalen Anpassungen ist weder als Herstellerprodukt noch als Inhouse-System lesbar, und keine Dokumentation beschreibt die Kombination.

## Causes ▼

- [Übermäßige Anpassung](uebermaessige-anpassung.md)
<br/>  Wenn das Ausmaß der erforderlichen Anpassung übersteigt, was die Erweiterungsmechanismen bequem unterstützen, wird das Bearbeiten des Kerns zum Weg des geringsten Widerstands.
- [Marktdruck](marktdruck.md)
<br/>  Eine bei der Beschaffung oder einem Deal eingegangene Verpflichtung muss bis zu einem Termin eingehalten werden, und der unterstützte Weg ist langsamer, als der Termin erlaubt.
- [Übermäßige Gefälligkeit gegenüber Stakeholdern](uebermaessige-gefaelligkeit-gegenueber-stakeholdern.md)
<br/>  Anfragen werden akzeptiert, ohne zu bewerten, ob sie im Standard erfüllt werden können, und die technische Konsequenz wird erst später entdeckt.
- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Niemand stellte fest, ob die Anforderung durch Konfiguration erfüllt werden könnte, sodass die Entwicklung dort begann, wo der Code sich zufällig befand.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Mitarbeitende, die das Erweiterungs-Framework des Produkts kennen, sind rar, während jeder eine Routine bearbeiten kann, sodass der nicht unterstützte Weg auch der verfügbare ist.
- [Vakuum an Projektautorität](vakuum-an-projektautoritaet.md)
<br/>  Niemand hat die Stellung, eine Modifikation abzulehnen oder auf dem unterstützten Mechanismus zu bestehen, sodass die Entscheidung von demjenigen getroffen wird, der gerade implementiert.

## Detection Methods ○

- Nutzung der eigenen Werkzeuge des Produkts, um Objekte aufzulisten, die vom gelieferten Zustand abweichen; die meisten Enterprise-Produkte können dies direkt melden
- Vergleich der installierten Codebasis mit einer sauberen Installation derselben Version und Zählung der abweichenden Objekte
- Überprüfung der Aufwandsaufteilung des letzten Upgrades und Identifikation, wie viel davon auf den Abgleich von Modifikationen entfiel statt auf Testen oder Schulung
- Prüfung, ob Support-Anfragen mit der Begründung abgelehnt oder eingeschränkt wurden, dass das System modifiziert ist
- Zählung, wie viele modifizierte Objekte keinen dokumentierten Grund, keinen Verantwortlichen und keinen Test haben
- Die Frage, ob jemand innerhalb eines Tages eine Liste jeder Modifikation und ihres Grundes erstellen kann

## Examples

Ein Hersteller, der ein ERP-Produkt betrieb, hatte über vierzehn Jahre 340 gelieferte Objekte modifiziert. Die meisten Modifikationen waren klein – ein zusätzliches Feld auf einem Bildschirm, eine zusätzliche Validierung, eine geänderte Sortierreihenfolge – und jede war zum jeweiligen Zeitpunkt die vernünftige Wahl gewesen. Der kumulative Effekt war, dass ein Herstellerrelease, das der Hersteller als Routine-Update beschrieb, das Team fünf Monate kostete, wovon etwa vier für den Abgleich von Modifikationen und das Regressionstesten des Ergebnisses aufgewendet wurden. Sie waren vier Hauptversionen zurück. Zwei der Modifikationen implementierten bei genauerer Untersuchung Verhalten, das das Standardprodukt sechs Jahre zuvor in einem Release erhalten hatte, sodass der Fork gepflegt wurde, um eine schlechtere Version eines Features zu bewahren, das der Hersteller inzwischen auslieferte.

Ein anderes Muster zeigte sich bei einem Dokumentenmanagement-Deployment. Die Organisation hatte die gelieferte Aufbewahrungsroutine modifiziert, um eine Regel spezifisch für eine Abteilung unterzubringen. Jahre später erforderte eine regulatorische Änderung eine Anpassung der Aufbewahrungsbehandlung, die der Hersteller als Patch auslieferte. Das Anwenden hätte die lokale Regel entfernt; das Nicht-Anwenden ließ die Organisation nicht konform. Keine Option war ohne ein Projekt verfügbar, und die Abteilung, deren Regel die ursprüngliche Modifikation ausgelöst hatte, war vier Jahre zuvor wegorganisiert worden – eine Tatsache, die niemand feststellte, bis der Abgleich jemanden zwang zu fragen, wer das Verhalten noch benötigte.
