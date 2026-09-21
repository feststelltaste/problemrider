---
title: Gefangenschaft durch Anbieterabhängigkeit
description: Legacy-Systeme werden durch eingestellte Anbieterprodukte gefangen,
  was teure individuelle Support-Verträge oder vollständigen Systemersatz erzwingt.
category:
- Code
- Management
related_problems:
- slug: vendor-lock-in
  similarity: 0.8
- slug: vendor-dependency
  similarity: 0.75
- slug: dependency-on-supplier
  similarity: 0.75
- slug: implementation-partner-dependency
  similarity: 0.7
- slug: legacy-skill-shortage
  similarity: 0.65
- slug: voided-vendor-support
  similarity: 0.65
solutions:
- anti-corruption-layer
- dependency-management-strategy
- data-export
- multi-cloud-iac
- platform-independence
- platform-independent-data-storage
- platform-independent-programming-languages
- vendor-management-practice
- risk-quantification
- modernization-options-comparison
- cost-of-delay
- system-decommissioning
- continuous-dependency-updates
- automated-code-migration
- retention-and-disposal-policy
layout: problem
lang: de
en_slug: vendor-dependency-entrapment
---

## Description

Gefangenschaft durch Anbieterabhängigkeit tritt auf, wenn Legacy-Systeme kritisch abhängig von Anbieterprodukten, -plattformen oder -diensten werden, die eingestellt wurden, nicht mehr unterstützt werden oder im End-of-Life-Status sind. Dies schafft eine schwerwiegendere Situation als typisches Vendor Lock-in, weil der Anbieter bereits strategische Entscheidungen getroffen hat, die zukünftige Support-Optionen einschränken oder eliminieren. Organisationen stehen vor unmöglichen Entscheidungen zwischen der Zahlung eskalierender Kosten für individuellen Support, der Akzeptanz zunehmender Sicherheits- und operativer Risiken oder der Durchführung teurer Notfall-Systemersätze.

## Indicators ⟡

- Anbieterankündigungen über Produkteinstellung oder End-of-Life-Zeitpläne für kritische Systemkomponenten
- Support-Verträge, die mit reduzierten Serviceniveaus zunehmend teurer werden
- Anbieterkonsolidierung oder -übernahme, die zu Änderungen der Produktstrategie führt
- Sicherheitspatches oder Updates, die für kritische Systemkomponenten nicht mehr bereitgestellt werden
- Anbieter-Vertriebsteams, die zur Migration zu neueren Produkten drängen, während sie Support für bestehende reduzieren
- Drittanbieter-Wartungsanbieter als einzige Option für fortgesetzten Systemsupport
- Hardware- oder Softwarekomponenten, die vom ursprünglichen Anbieter nicht mehr hergestellt oder entwickelt werden

## Symptoms ▲

- [Anstieg der Wartungskosten](anstieg-der-wartungskosten.md)
<br/>  Individuelle Support-Verträge für eingestellte Produkte werden zunehmend teurer, während weniger Spezialisten verfügbar bleiben.
- [Technologie-Isolation](technologie-isolation.md)
<br/>  Das System wird auf eingestellter Technologie isoliert, die sich nicht mit modernen Werkzeugen und Plattformen integrieren kann.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Während Anbieterprodukte eingestellt werden, pflegen weniger Fachleute Fähigkeiten in diesen Technologien, was Talente knapp macht.

## Causes ▼

- [Vendor Lock-in](vendor-lock-in.md)
<br/>  Tiefe Integration mit anbieterspezifischen Technologien macht es unmöglich, sich anzupassen, wenn der Anbieter Produkte einstellt.
- [Anbieterabhängigkeit](anbieterabhaengigkeit.md)
<br/>  Exzessives Vertrauen auf einen einzelnen Anbieter schafft Verwundbarkeit, wenn dieser Anbieter die Strategie ändert oder Produkte einstellt.

## Detection Methods ○

- Überwachung von Anbieter-Produkt-Roadmaps und End-of-Life-Ankündigungen für alle kritischen Systemabhängigkeiten
- Verfolgung von Anbieter-Support-Vertragskosten und Serviceniveauänderungen über die Zeit
- Bewertung der Systemarchitektur auf einzelne Punkte der Anbieterabhängigkeit
- Bewertung der finanziellen Gesundheit und Marktposition des Anbieters auf Anzeichen von Geschäftsrisiko
- Überprüfung von Anbieter-Support-Vorfällen und Antwortzeiten auf Verschlechterungsmuster
- Durchführung regelmäßiger Anbieter-Risikobewertungen einschließlich Support-Fortsetzungsszenarien
- Überwachung von Industrietrends und Anbieterkonsolidierung, die Support-Verfügbarkeit beeinflussen könnten
- Bewertung der technischen Machbarkeit und Kosten der Migration weg von aktuellen Anbieterabhängigkeiten

## Examples

Eine Gesundheitsorganisation baute ihr Patientenaktensystem vor 12 Jahren auf einer spezialisierten Datenbankplattform eines mittelgroßen Softwareanbieters auf. Der Anbieter wurde von einem größeren Unternehmen übernommen, das die Einstellung des Datenbankprodukts zugunsten seiner eigenen konkurrierenden Lösung ankündigte. Die Gesundheitsorganisation steht vor drei schwierigen Optionen: 500.000 $ jährlich für individuellen Support vom verbleibenden Personal des ursprünglichen Anbieters zahlen (ohne Garantie langfristiger Verfügbarkeit), zur Datenbank des übernehmenden Unternehmens migrieren (18 Monate und 3 Millionen $ zum Umschreiben aller Anwendungen erfordernd), oder zu einem völlig anderen Anbieter migrieren (24 Monate und 5 Millionen $ für eine komplette Systemüberholung erfordernd). Während des Entscheidungsprozesses wird eine kritische Sicherheitslücke in der Datenbank entdeckt, aber es wird kein Patch entwickelt, weil das Produkt eingestellt ist. Die Organisation muss teure Netzwerkisolation und Monitoring implementieren, um das Sicherheitsrisiko zu mindern, während sie ihre Migration plant. Die Situation zwingt sie, zwischen operativem Risiko, massiven unerwarteten Ausgaben oder Geschäftsstörung durch ein Notfall-Systemersatzprojekt zu wählen, alles weil ihre Anbieterabhängigkeit zu einer strategischen Belastung wurde, die sie nicht kontrollieren können.
