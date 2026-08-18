---
title: Aufbewahrungspflichten blockieren Änderungen
description: Gesetzliche Aufbewahrungspflichten hängen an Daten, deren Format und
  System niemand ändern kann, sodass die Pflicht das System einfriert, das sie hält.
category:
- Database
- Operations
- Security
related_problems:
- slug: schema-evolution-paralysis
  similarity: 0.55
- slug: regulatory-compliance-drift
  similarity: 0.55
- slug: data-migration-integrity-issues
  similarity: 0.55
- slug: data-migration-complexities
  similarity: 0.5
- slug: legacy-system-documentation-archaeology
  similarity: 0.5
- slug: system-stagnation
  similarity: 0.5
solutions:
- retention-and-disposal-policy
- data-archiving
- audit-trail-management
- system-decommissioning
- datensparsamkeit
- risk-quantification
- application-portfolio-inventory
- parallel-run
- checksums
- clear-ownership-model
layout: problem
lang: de
en_slug: retention-obligations-block-change
---

## Description

Aufbewahrungspflichten blockieren Änderungen, wenn Daten, die eine Organisation gesetzlich für Jahre oder Jahrzehnte aufbewahren muss, in einem System gehalten werden, das nicht modifiziert, migriert oder abgeschaltet werden kann, ohne diese Pflicht zu gefährden. Die Anforderung besteht meist nicht nur darin, dass die Daten weiter existieren, sondern dass sie abrufbar, lesbar, vollständig und nachweislich unverändert bleiben. Diese Kombination macht Migration weit schwieriger als bloßes Verschieben von Datensätzen: Die Organisation muss zeigen können, dass das, was sie Jahre später produziert, dem ursprünglich Erfassten entspricht. Angesichts dieser Bürde und einer unklaren rechtlichen Grenze ist die sichere Antwort stets, nichts zu ändern, und das System, das die Daten hält, friert ein. Da niemand festgelegt hat, was tatsächlich aufbewahrt werden muss und wie lange, erstreckt sich das Einfrieren auf alles statt auf die tatsächlich betroffene Teilmenge.

## Indicators ⟡

- Die Aufbewahrungsfrist für die Daten wird in Jahren angegeben, und niemand kann die Quellpflicht vorlegen
- Das System kann wegen aufbewahrter Daten nicht abgeschaltet werden, und es existiert kein Plan für die Daten selbst
- Alte Instanzen werden allein am Laufen gehalten, damit Datensätze abrufbar bleiben
- Niemand kann sagen, welcher Anteil der aufbewahrten Daten tatsächlich einer Pflicht unterliegt
- Löschung wurde nie durchgeführt, und es existiert kein Prozess, durch den sie erfolgen würde
- Rechts- und Technikpersonal haben nie gemeinsam untersucht, was die Pflicht tatsächlich erfordert

## Symptoms ▲

- [Veraltete Technologien](veraltete-technologien.md)
<br/>  Systeme werden über ihre unterstützte Lebensdauer hinaus am Laufen gehalten, rein als Datenverwahrer, zusammen mit den Laufzeitumgebungen und der Hardware, die sie benötigen.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Lizenzen, Infrastruktur, Patching und Monitoring laufen weiter für Systeme, die keinem operativen Zweck mehr dienen.
- [Lähmung der Modernisierungsstrategie](laehmung-der-modernisierungsstrategie.md)
<br/>  Jede Option für das System scheitert an den aufbewahrten Daten, und weil die Pflicht ungeprüft ist, kann keine Option ordentlich bewertet werden.
- [Komplexität der Datenmigration](komplexitaet-der-datenmigration.md)
<br/>  Die Migration aufbewahrter Datensätze erfordert den Nachweis, dass Bedeutung und Integrität erhalten bleiben, was eine weit stärkere Anforderung ist als sie nur zu verschieben.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Systeme, die zur Aufbewahrung am Laufen gehalten werden, erfordern Fähigkeiten, die niemand mehr erwirbt, und der Pool schrumpft über die gesamte Aufbewahrungsdauer.
- [Gefangenschaft durch Anbieterabhängigkeit](gefangenschaft-durch-anbieterabhaengigkeit.md)
<br/>  Ein eingefrorenes System kann nicht ersetzt oder neu verhandelt werden, was jede kommerzielle Option gegenüber seinem Lieferanten beseitigt.

## Causes ▼

- [Regulatorische Compliance-Drift](regulatorische-compliance-drift.md)
<br/>  Pflichten wurden nie auf spezifische Daten abgebildet, sodass das gesamte System statt des tatsächlich betroffenen Anteils als abgedeckt behandelt wird.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Aufbewahrung sitzt zwischen Recht, Betrieb und Geschäft, und keiner von ihnen besitzt die Verantwortung festzulegen, was tatsächlich aufbewahrt werden muss.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Die Konsequenz, Aufbewahrung falsch zu handhaben, ist rechtlicher statt technischer Natur, sodass die risikoscheue Antwort ist, alles einzufrieren.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Die Bedeutung aufbewahrter Datensätze hängt von Strukturen und Codes ab, die nie dokumentiert wurden, sodass niemand behaupten kann, dass eine migrierte Kopie dasselbe bedeutet.
- [Übermäßiger Einsatz von Entity-Attribute-Value](uebermaessiger-einsatz-von-entity-attribute-value.md)
<br/>  Wo die gespeicherte Form untypisiert und nur konventionell selbstbeschreibend ist, wird der Nachweis, dass eine Migration die Bedeutung erhalten hat, sehr schwierig.

## Detection Methods ○

- Fragen Sie nach der spezifischen rechtlichen Quelle jeder genutzten Aufbewahrungsfrist und wie viele davon vorgelegt werden können
- Messen Sie, welcher Anteil aufbewahrter Daten innerhalb einer Pflichtperiode liegt und welcher Anteil abgelaufen ist
- Stellen Sie fest, ob jemals eine Löschung stattgefunden hat und welcher Prozess dafür genutzt würde
- Zählen Sie Systeme, die allein zur Datenaufbewahrung laufen, und summieren Sie deren jährliche Kosten
- Prüfen Sie, ob Rechts- und Technikpersonal gemeinsam bewertet haben, was abrufbar, lesbar und unverändert in der Praxis erfordern
- Testen Sie den Abruf eines Datensatzes aus der ältesten aufbewahrten Periode und erfassen Sie, wie lange es dauert und was erforderlich ist

## Examples

Ein Versicherer hielt drei abgelöste Policenverwaltungssysteme allein deshalb am Laufen, weil Policendokumente für Zeiträume bis zu dreißig Jahren nach Vertragsende abrufbar bleiben mussten. Die kombinierten jährlichen Kosten für Lizenzen, Infrastruktur und den Spezialistenvertrag, der zum Betrieb eines der Systeme gehalten wurde, waren beträchtlich und waren neun Jahre lang ohne Prüfung verlängert worden. Eine gemeinsame Überprüfung durch Recht und Technik ergab, dass die Pflicht am Policendokument und einer definierten Menge von Transaktionsdatensätzen hing, nicht am operativen System, und dass ein Archiv, das diese Artefakte mit einer Integritätsgarantie bewahrt, dies erfüllen würde. Zwei der drei Systeme wurden innerhalb eines Jahres abgeschaltet.

Dieselbe Überprüfung fand daneben das gegenteilige Problem. Etwa 40 Prozent der aufbewahrten Daten waren über jede geltende Frist hinaus und hätten Jahre zuvor gelöscht werden sollen — was nicht nur eine Speicherfrage war, da die Aufbewahrung personenbezogener Daten über ihre rechtmäßige Frist hinaus in der geltenden Rechtsordnung selbst einen Verstoß darstellt. Die Organisation hatte ein Jahrzehnt lang angenommen, dass Aufbewahrung eine Frage des Behaltens sei, und nie in Betracht gezogen, dass die Pflicht sowohl eine Obergrenze als auch eine Untergrenze hatte. Niemand war dafür verantwortlich gewesen zu fragen, weil Aufbewahrung von der Technik als Rechtsthema und vom Recht als Technikthema behandelt worden war.
