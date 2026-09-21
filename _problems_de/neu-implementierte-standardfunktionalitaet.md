---
title: Neu implementierte Standardfunktionalität
description: Eine Fähigkeit, die das Produkt bereits bietet, wurde als individuelle
  Entwicklung neu gebaut, was die Wartungslast erhöht, während der Nutzen des Produktkaufs
  entfällt.
category:
- Architecture
- Process
- Business
related_problems:
- slug: excessive-customization
  similarity: 0.7
- slug: process-software-misfit
  similarity: 0.65
- slug: implementation-rework
  similarity: 0.65
- slug: core-modification-of-standard-software
  similarity: 0.65
- slug: customization-outside-version-control
  similarity: 0.6
- slug: implementation-partner-dependency
  similarity: 0.6
solutions:
- fit-to-standard-principle
- functional-gap-analysis
- standard-software
- variant-consolidation
- customization-cost-attribution
- feature-usage-measurement
- domain-immersion
- lightweight-design-review
- technology-radar
- strategic-code-deletion
layout: problem
lang: de
en_slug: reimplemented-standard-functionality
---

## Description

Neu implementierte Standardfunktionalität tritt auf, wenn eine Organisation als individuelle Entwicklung innerhalb eines kommerziell erworbenen Softwaresystems eine Fähigkeit baut, die das Produkt bereits anbietet. Es geschieht, weil niemand festgestellt hat, was der Standard leisten könnte, bevor entschieden wurde, ihn zu bauen. Die Gründe sind banal: Die Dokumentation des Produkts ist umfangreich und unbekannt, die Standardimplementierung sieht leicht anders aus als angefordert, eine externe Beratung verdient mehr an Entwicklung als an Konfiguration, oder die Anforderung kam als Lösung formuliert statt als Bedürfnis an. Die Organisation zahlt dann zweimal – den Kaufpreis für Funktionalität, die sie nicht nutzt, und die laufende Wartung für eine Version, die sie selbst funktionsfähig halten muss. Die maßgeschneiderte Implementierung hinkt typischerweise auch hinterher: Der Anbieter verbessert das Standard-Feature über die Jahre, während die lokale Kopie bleibt, wie sie geschrieben wurde.

## Indicators ⟡

- Eine Anbieter-Release-Notiz beschreibt ein Feature, das Sie bereits haben, und niemand ist sich sicher, ob gewechselt werden soll
- Berater schlagen Entwicklung für Anforderungen vor, die generisch statt spezifisch für Ihr Geschäft klingen
- Niemand im Team kann sagen, was das Standardprodukt in einem Bereich tut, ohne es zu öffnen und nachzusehen
- Individuelle Entwicklungen tragen Namen, die eng den Namen der Standardmodule ähneln
- Schulungsmaterial für das Standardprodukt beschreibt Bildschirme, die Ihre Nutzer nie sehen
- Eine Anforderung wurde wie vom Anfragenden formuliert umgesetzt, statt zu fragen, welches Problem sie löste

## Symptoms ▲

- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Die Organisation wartet Funktionalität, die sie als Teil des Produkts hätte erhalten können, einschließlich der Aufrechterhaltung über Upgrades hinweg.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Aufwand wird darauf verwendet, Fähigkeiten zu bauen und dann zu erweitern, die bereits gekauft und verfügbar waren.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Der ursprüngliche Bau produzierte nichts, was die Organisation nicht bereits hatte, und seine Kosten werden selten als Verschwendung erkannt, weil das Ergebnis funktioniert.
- [Veraltete Technologien](veraltete-technologien.md)
<br/>  Individuelle Implementierungen profitieren nicht von den Verbesserungen des Anbieters, sodass die lokale Version hinter dem Standard zurückbleibt, den sie ersetzt hat.
- [Testkomplexität](testkomplexitaet.md)
<br/>  Individuelle Fähigkeiten müssen bei jedem Upgrade regressionsgetestet werden, während Standardfähigkeiten vom Anbieter getestet werden.
- [Schwierigkeiten beim Quantifizieren von Nutzen](schwierigkeiten-beim-quantifizieren-von-nutzen.md)
<br/>  Die Verschwendung ist unsichtbar, weil die individuelle Implementierung funktioniert, sodass nichts jemanden dazu veranlasst, sie mit dem Standard zu vergleichen.
- [Übermäßige Anpassung](uebermaessige-anpassung.md)
<br/>  Jede Neuimplementierung trägt zum Volumen des lokalen Codes bei, den jede zukünftige Änderung und jedes Upgrade berücksichtigen muss.

## Causes ▼

- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Anforderungen werden wie angefragt aufgezeichnet statt untersucht, sodass niemand fragt, ob das zugrunde liegende Bedürfnis bereits erfüllt ist.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Niemand in der Organisation kennt das Produkt tief genug, um zu erkennen, dass die angeforderte Fähigkeit bereits existiert.
- [Abhängigkeit vom Zulieferer](abhaengigkeit-vom-zulieferer.md)
<br/>  Ein für Entwicklung bezahlter Implementierungspartner hat keinen Anreiz darauf hinzuweisen, dass Konfiguration ausreichen würde, und weiß es oft auch nicht.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Produktfähigkeit wird vom Anbieter in Umfängen dokumentiert, die niemand liest, sodass das, was der Standard bietet, intern effektiv unbekannt ist.
- [Marktdruck](marktdruck.md)
<br/>  Unter Zeitdruck ist der Bau des Bekannten vorhersehbarer als die Untersuchung, ob ein unbekanntes Standard-Feature passt.
- [Übermäßige Gefälligkeit gegenüber Stakeholdern](uebermaessige-gefaelligkeit-gegenueber-stakeholdern.md)
<br/>  Ein Anfragender, der die gewünschte Lösung beschreibt, bekommt sie gebaut, statt gefragt zu werden, welches Ergebnis er benötigt.

## Detection Methods ○

- Für jede substanzielle individuelle Entwicklung fragen, welche Standardfähigkeit bewertet und abgelehnt wurde, und warum; das Fehlen einer Antwort ist der Befund
- Vergleich des Bestands individueller Entwicklungen mit der Modul- und Feature-Liste des Produkts, auf Überlappung im Zweck achtend
- Überprüfung der Anbieter-Release-Notizen der letzten mehreren Jahre auf Features, die etwas duplizieren, das Sie warten
- Den Anbieter oder einen unabhängigen Experten bitten, den individuellen Bestand zu überprüfen; dies ist ein Service, den die meisten Anbieter anbieten
- Prüfen, ob eine Bewertung des Standards den letzten fünf Entwicklungsentscheidungen vorausging
- Suche nach individuellen Entwicklungen, deren funktionale Beschreibung gleichermaßen für jede Organisation in Ihrer Branche gelten würde

## Examples

Eine Organisation des öffentlichen Sektors, die eine Dokumentenmanagementplattform betreibt, hatte während der Implementierung einen individuellen Genehmigungsworkflow in Auftrag gegeben, zu Kosten von etwa neun Monaten Beratung. Sieben Jahre später fand ein neuer Administrator, der sich durch das Schulungsmaterial des Produkts arbeitete, heraus, dass das Standardprodukt eine äquivalente Workflow-Fähigkeit in dem Release vor demjenigen ausgeliefert hatte, das sie implementiert hatten. Der individuelle Workflow war gebaut worden, weil die Anforderung, wie geschrieben, eine zweistufige Genehmigung mit einer Delegationsregel spezifizierte – und die Standardfähigkeit unterstützte beides, konfiguriert statt codiert. Niemand hatte nachgesehen. Die Migration zum Standard dauerte sechs Wochen und entfernte eine Komponente, die bei jedem Upgrade sieben Jahre lang Regressionstest-Aufwand verbraucht hatte.

Eine Enterprise-Resource-Planning-Implementierung zeigte das Muster in kleinerem Maßstab, aber höherer Häufigkeit. Eine Überprüfung von 61 individuellen Entwicklungen gegen die Fähigkeitsliste des Produkts fand 14, die Standardfunktionalität direkt duplizierten, und weitere 9, die sie mit einer kleinen Variation duplizierten. Der lehrreichste war ein individueller Bericht, der die Bestandsbewertung neu berechnete: Der Standardbericht produzierte dieselben Zahlen, formatierte die Währung aber anders, und 2013 hatte ein Controller gebeten, das Format zu ändern. Statt des Formats war die Berechnung neu gebaut worden.
