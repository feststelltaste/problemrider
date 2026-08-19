---
title: Upgrade durch Anpassungen blockiert
description: Anbieter-Releases können nicht angewendet werden, weil die angehäufte
  lokale Anpassung jedes Mal abgeglichen und revalidiert werden müsste.
category:
- Dependencies
- Operations
- Process
related_problems:
- slug: core-modification-of-standard-software
  similarity: 0.7
- slug: excessive-customization
  similarity: 0.65
- slug: vendor-dependency-entrapment
  similarity: 0.65
- slug: voided-vendor-support
  similarity: 0.6
- slug: reimplemented-standard-functionality
  similarity: 0.55
- slug: schema-evolution-paralysis
  similarity: 0.55
solutions:
- fit-to-standard-principle
- explicit-extension-points
- continuous-dependency-updates
- characterization-tests
- automated-tests
- regression-testing
- parallel-run
- staged-investment-with-decision-gates
- cost-of-delay
- risk-quantification
- variant-consolidation
- no-regret-moves
- modernization-options-comparison
- customization-under-version-control
layout: problem
lang: de
en_slug: upgrade-blocked-by-customization
---

## Description

Upgrade-Blockade tritt auf, wenn der Aufwand, eine Installation auf ein neues Anbieter-Release zu bringen, das übersteigt, was die Organisation bereit ist auszugeben, sodass Releases übersprungen werden. Jedes übersprungene Release macht das nächste schwieriger, weil der Abgleich nun mehrere Versionen von Anbieteränderungen gleichzeitig umfasst. Der Zustand verstärkt sich, bis die installierte Version den Anbieter-Support verlässt, zu welchem Zeitpunkt Sicherheitspatches aufhören, der verfügbare Fähigkeitenpool schrumpft und das schließliche Upgrade kein Upgrade mehr ist, sondern eine Migration. Was dies von gewöhnlichem Aufschub unterscheidet, ist, dass die Blockade selbstverschuldet und kumulativ ist: Die Organisation wartet auf nichts Externes, und jeder Monat des Wartens erhöht den Preis der Entscheidung, die sie vermeidet.

## Indicators ⟡

- Die installierte Version ist mehr als ein Hauptrelease zurück, und die Lücke hat sich über die Zeit vergrößert
- Der Upgrade-Aufwand wird in Monaten geschätzt, und die Schätzung ist zwischen aufeinanderfolgenden Versuchen gewachsen
- Ein Upgrade wurde mindestens einmal geplant und abgesagt
- Antworten des Anbieter-Supports beginnen zunehmend damit, Sie zu bitten, das Problem auf einer aktuellen Version zu reproduzieren
- Niemand kann den Gesamtaufwand nennen, ohne eine Discovery-Übung durchzuführen, die Wochen kostet
- Neue Fähigkeit, die das Geschäft möchte, ist in einem Release verfügbar, das die Organisation nicht erreichen kann

## Symptoms ▲

- [Veraltete Technologien](veraltete-technologien.md)
<br/>  Die Installation altert aus dem Anbieter-Support hinaus, und die Laufzeitumgebung, Datenbank und das Betriebssystem, von denen sie abhängt, altern mit ihr.
- [Regulatorische Compliance-Drift](regulatorische-compliance-drift.md)
<br/>  Regulatorische Änderungen, die der Anbieter als Produktupdates liefert, können nicht empfangen werden, sodass Compliance lokal gebaut werden muss oder schlicht fehlt.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Der Pool an Personen, die eine nicht unterstützte Version kennen, schrumpft kontinuierlich, und neue Mitarbeiter haben keinen Grund, sie gelernt zu haben.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Fähigkeit, die der Anbieter bereits gebaut hat und für die die Organisation bereits bezahlt hat, bleibt für Nutzer unbegrenzt nicht verfügbar.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Fehler, die der Anbieter behoben hat, müssen lokal umgangen werden, und erweiterte Support-Verträge für alte Versionen tragen einen Aufpreis.
- [Scheiternde ROI-Rechtfertigung für Modernisierung](scheiternde-roi-rechtfertigung-fuer-modernisierung.md)
<br/>  Das Upgrade wächst zu einer Zahl, die groß genug ist, dass kein Business Case erfolgreich ist, was weiteren Aufschub garantiert.
- [Gefangenschaft durch Anbieterabhängigkeit](gefangenschaft-durch-anbieterabhaengigkeit.md)
<br/>  Eine auf einer nicht unterstützten Version gestrandete Installation hat weder einen tragfähigen Upgrade-Pfad noch einen tragfähigen Ersatzpfad.

## Causes ▼

- [Kernmodifikation von Standardsoftware](kernmodifikation-von-standardsoftware.md)
<br/>  Modifizierte Anbieterobjekte müssen gegen jedes Release abgeglichen werden, was die einzelgrößte Komponente der meisten Upgrade-Schätzungen ist.
- [Übermäßige Anpassung](uebermaessige-anpassung.md)
<br/>  Das Volumen der lokalen Anpassung bestimmt, wie viel revalidiert werden muss, und es wächst kontinuierlich, während das Upgrade aufgeschoben wird.
- [Anpassungen außerhalb der Versionskontrolle](anpassungen-ausserhalb-der-versionskontrolle.md)
<br/>  Wo das Anpassungsinventar nicht aufgelistet werden kann, kann das Upgrade nicht abgegrenzt werden, sodass Schätzungen groß und defensiv sind.
- [Erhöhter manueller Testaufwand](erhoehter-manueller-testaufwand.md)
<br/>  Ohne automatisierte Regressionsabdeckung ist die Revalidierung einer vollständigen Produktinstallation eine manuelle Übung, gemessen in Personenmonaten.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  In jedem gegebenen Quartal ist das Upgrade weniger dringend als das, was gerade geliefert wird, und dieser Vergleich wird wiederholt mit demselben Ergebnis angestellt.

## Detection Methods ○

- Erfassen Sie die installierte Version, das aktuelle Release und das Datum, an dem der Support für das, was Sie betreiben, endet
- Zeichnen Sie die Lücke zwischen installierter und aktueller Version über die letzten fünf Jahre auf; der Trend zählt mehr als das Niveau
- Vergleichen Sie die Aufwandsschätzungen aufeinanderfolgender Upgrade-Versuche, um festzustellen, ob die Abgleichslast wächst
- Messen Sie, welcher Anteil des letzten Upgrades in den Abgleich von Modifikationen ging im Vergleich zu Testen und Training
- Zählen Sie, wie viele vom Anbieter gelieferte Fixes lokal neu implementiert wurden, weil das Release, das sie trug, nicht angewendet werden konnte
- Fragen Sie, was zutreffen müsste, damit ein Upgrade routinemäßig ist, und behandeln Sie die Lücke zwischen dem und der Realität als den tatsächlichen Rückstand

## Examples

Eine ERP-Installation (Enterprise Resource Planning) war zuletzt vor sechs Jahren aktualisiert worden. Zwei nachfolgende Versuche waren geplant und abgesagt worden: der erste, als Discovery den Aufwand auf acht Monate gegenüber einem Vier-Monats-Budget feststellte, der zweite, als die Schätzung der Beratungsfirma höher zurückkam als die erste. Beim dritten Versuch hatte die installierte Version den Mainstream-Support verlassen, die Organisation zahlte einen Aufpreis für erweiterten Support, und eine regulatorische Änderung der Rechnungsstellung musste lokal implementiert werden, weil sie als Anbieter-Update ankam, das sie nicht anwenden konnten. Das schließliche Programm dauerte vierzehn Monate. Etwa zwei Drittel des Aufwands gingen in 340 modifizierte Anbieterobjekte und in Regressionstests, denen jegliche automatisierte Grundlage fehlte.

Die Verstärkung war in den Zahlen sichtbar. Der erste abgesagte Versuch hatte acht Monate geschätzt. Vier Jahre und etwa 60 weitere Anpassungen später wurde derselbe Umfang auf dreizehn geschätzt. Nichts Externes hatte sich geändert. Die Organisation hatte vier Jahre damit verbracht, die Entscheidung, die sie vermied, teurer zu machen, und zu keinem Zeitpunkt hatte jemand berechnet, was der Aufschub pro Monat kostete — was, als es schließlich während des dritten Versuchs berechnet wurde, die jährlichen Kosten des erweiterten Support-Vertrags überstieg, der als der Preis des Wartens behandelt worden war.
