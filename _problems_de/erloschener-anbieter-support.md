---
title: Erloschener Anbieter-Support
description: Der Anbieter lehnt die Diagnose von Problemen ab, weil die Installation
  modifiziert ist, sodass die Organisation für Support bezahlt, den sie nicht mehr
  nutzen kann.
category:
- Dependencies
- Operations
- Business
related_problems:
- slug: core-modification-of-standard-software
  similarity: 0.65
- slug: vendor-dependency-entrapment
  similarity: 0.65
- slug: upgrade-blocked-by-customization
  similarity: 0.6
- slug: reimplemented-standard-functionality
  similarity: 0.55
- slug: implementation-partner-dependency
  similarity: 0.55
- slug: excessive-customization
  similarity: 0.55
solutions:
- vendor-management-practice
- explicit-extension-points
- fit-to-standard-principle
- service-level-agreements
- risk-quantification
- total-cost-of-ownership-transparency
- application-portfolio-inventory
- knowledge-rotation
- written-first-communication
layout: problem
lang: de
en_slug: voided-vendor-support
---

## Description

Erloschener Anbieter-Support tritt auf, wenn lokale Modifikation eines kommerziellen Softwareprodukts der Organisation den praktischen Zugang zu dem Support entzieht, für den sie bezahlt. Die Ablehnung ist selten absolut. Häufiger ist sie prozedural: Der Anbieter bittet darum, das Problem auf einer unmodifizierten Installation zu reproduzieren, was die Organisation nicht bereitstellen kann, oder er beschränkt seine Verantwortung auf den gelieferten Code, was nicht das ist, was läuft. Der Effekt ist derselbe. Die Organisation zahlt weiterhin eine Support-Gebühr, und jeder Vorfall wird intern von Personen mit weit weniger Produktwissen diagnostiziert als der Anbieter hat. Weil die Ablehnung pro Vorfall geschieht statt als formaler Rückzug, wird die Position oft nicht als eine Entscheidung erkannt, die jemand getroffen hat — sie hat sich angehäuft.

## Indicators ⟡

- Support-Tickets werden routinemäßig mit der Bitte geschlossen, auf einem Standardsystem zu reproduzieren
- Das Team eröffnet für bestimmte Module keine Anbieter-Tickets mehr, weil das Ergebnis vorhersehbar ist
- Vorfälle in modifizierten Bereichen werden unabhängig von der Schwere vollständig intern diagnostiziert
- Niemand kann sagen, was der Support-Vertrag angesichts des aktuellen Installationszustands tatsächlich abdeckt
- Die Verlängerung wird jährlich genehmigt, ohne dass jemand bewertet, welcher Wert erzielt wurde
- Eskalationen erfordern den Account Manager statt des Support-Prozesses

## Symptoms ▲

- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Probleme, die der Anbieter aus Erfahrung diagnostizieren könnte, werden lokal von Grund auf erarbeitet, was weit länger dauert.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Die Organisation muss intern tiefes Produktwissen in einem Produkt aufrechterhalten, das sie nicht geschrieben hat, weil sie nicht mehr auf das des Anbieters zurückgreifen kann.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Eine Support-Gebühr wird für einen Service bezahlt, der nicht genutzt werden kann, während die Diagnosearbeit, die er abdecken sollte, zu internen Kosten erledigt wird.
- [Ständiges Feuerlöschen](staendiges-feuerloeschen.md)
<br/>  Vorfälle in modifizierten Bereichen verbrauchen unverhältnismäßigen Aufwand, und ihre Häufigkeit sinkt nicht, weil die zugrunde liegenden Ursachen nie ordentlich diagnostiziert werden.
- [Belastete Anbieterbeziehung](belastete-anbieterbeziehung.md)
<br/>  Wiederholte Ablehnungen produzieren eine feindselige Dynamik, in der jede Seite die andere als in böser Absicht handelnd betrachtet.
- [Gefangenschaft durch Anbieterabhängigkeit](gefangenschaft-durch-anbieterabhaengigkeit.md)
<br/>  Die Organisation hängt von einem Produkt ab, dessen Hersteller ihr nicht helfen will, und kann nicht gehen, weil die Modifikationen neu gebaut werden müssten.

## Causes ▼

- [Kernmodifikation von Standardsoftware](kernmodifikation-von-standardsoftware.md)
<br/>  Modifikation gelieferten Codes ist die spezifische Bedingung, die die meisten Support-Vereinbarungen ausschließen, und sie ist üblicherweise, was die Ablehnung auslöst.
- [Upgrade durch Anpassungen blockiert](upgrade-durch-anpassungen-blockiert.md)
<br/>  Der Betrieb einer nicht mehr unterstützten Version entzieht den Anspruch vollständig, unabhängig von jeglicher Modifikation.
- [Schlechtes Vertragsdesign](schlechtes-vertragsdesign.md)
<br/>  Was der Anbieter angesichts einer modifizierten Installation unterstützen wird und was nicht, wird bei der Unterzeichnung häufig nicht angesprochen und während eines Vorfalls entdeckt.
- [Übermäßige Anpassung](uebermaessige-anpassung.md)
<br/>  Das Volumen der Anpassung macht es unpraktisch, irgendein Problem auf einer Standardinstallation zu reproduzieren, was der Anbieter verlangt.
- [Rechtsstreitigkeiten](rechtsstreitigkeiten.md)
<br/>  Wo die Beziehung strittig geworden ist, neigen beide Parteien dazu, die Support-Grenze eng auszulegen.

## Detection Methods ○

- Überprüfen Sie das letzte Jahr an Anbieter-Tickets und zählen Sie, wie viele ohne Diagnose aus Modifikations- oder Versionsgründen geschlossen wurden
- Fragen Sie das Support-Team, für welche Module sie keine Tickets mehr erstellen und warum
- Lesen Sie die Support-Vereinbarung gegen den aktuellen Installationszustand und identifizieren Sie, was tatsächlich abgedeckt ist
- Vergleichen Sie die Support-Gebühr mit dem erzielten Wert, gemessen als vom Anbieter erfolgreich gelöste Tickets
- Messen Sie die Vorfallslösungszeit in modifizierten versus unmodifizierten Bereichen des Produkts
- Fragen Sie, ob die Organisation ein gegebenes Problem auf einer sauberen Installation reproduzieren könnte, und wie lange das dauern würde

## Examples

Eine Organisation, die ein stark angepasstes ERP-System (Enterprise Resource Planning) betrieb, zahlte eine jährliche Support-Gebühr im hohen sechsstelligen Bereich. Eine Überprüfung der Tickets des vorangegangenen Jahres ergab, dass von 94 erstellten Tickets 61 mit der Bitte geschlossen worden waren, auf einem unmodifizierten System zu reproduzieren. Das Team hatte aufgehört, für drei Module überhaupt Tickets zu erstellen. Niemand hatte je die effektiven Kosten pro gelöstem Ticket berechnet, und als es berechnet wurde, war es etwa vierzigmal so viel, wie die Organisation angenommen hatte zu zahlen. Die Verlängerung war elf aufeinanderfolgende Jahre auf der Grundlage genehmigt worden, dass Support für ein kritisches System offensichtlich notwendig sei — was er war, und was nicht das war, was sie erhielten.

Eine Dokumentenmanagement-Bereitstellung zeigte, wie sich die Position anhäuft, statt gewählt zu werden. Eine einzelne Modifikation, die 2016 an einer Aufbewahrungsroutine vorgenommen wurde, stellte dieses Modul außerhalb des Supports. Dies wurde damals in einem Ticket-Kommentar vermerkt und nie eskaliert. In den folgenden Jahren wuchs der betroffene Bereich, während weitere Änderungen um die ursprüngliche herum vorgenommen wurden. Als ein Vorfall in diesem Bereich eine echte Compliance-Exposition verursachte, hatte sich die Grenze dessen, womit der Anbieter helfen würde, erheblich verschoben, und es existierte kein Dokument, das festhielt, wo sie nun lag oder wer ihr zugestimmt hatte.
