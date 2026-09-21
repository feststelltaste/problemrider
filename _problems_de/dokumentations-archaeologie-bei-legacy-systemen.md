---
title: Dokumentations-Archäologie bei Legacy-Systemen
description: Kritisches Systemwissen existiert nur in veralteten Dokumentationsformaten,
  überholten Diagrammen und dem Stammeswissen ausgeschiedener Mitarbeiter.
category:
- Communication
- Management
related_problems:
- slug: poor-documentation
  similarity: 0.65
- slug: legacy-business-logic-extraction-difficulty
  similarity: 0.65
- slug: legacy-skill-shortage
  similarity: 0.65
- slug: information-decay
  similarity: 0.65
- slug: implicit-knowledge
  similarity: 0.65
- slug: information-fragmentation
  similarity: 0.65
solutions:
- documentation-as-code
- knowledge-sharing-practices
- api-documentation
- architecture-documentation
- living-documentation
- requirements-traceability-matrix
- code-reading-sessions
- application-portfolio-inventory
- no-regret-moves
- baseline-measurement
- technical-debt-assessment
layout: problem
lang: de
en_slug: legacy-system-documentation-archaeology
---

## Description

Dokumentations-Archäologie bei Legacy-Systemen bezeichnet den herausfordernden Prozess, das Verständnis von Legacy-Systemen zu rekonstruieren, wenn kritisches Wissen nur in veralteten Formaten, überholter Dokumentation existiert oder mit ausgeschiedenen Mitarbeitern verloren gegangen ist. Dieses Problem erfordert Detektivarbeit, um Systemverhalten, Geschäftsregeln und architektonische Entscheidungen aus fragmentierten Quellen zusammenzusetzen, einschließlich alter Dokumente, Code-Kommentaren, Datenbankschemas und Interviews mit langjährigen Mitarbeitern, die möglicherweise unvollständige oder ungenaue Erinnerungen an Systemdetails haben.

## Indicators ⟡

- Systemdokumentation, die Jahre veraltet oder in obsoleten Formaten gespeichert ist
- Kritisches Systemwissen, konzentriert in den Erinnerungen weniger langjähriger Mitarbeiter
- Architekturdiagramme, die nicht mit aktuellem Systemverhalten oder -struktur übereinstimmen
- Geschäftsregeln, die von aktuellem Personal oder Dokumentation nicht erklärt werden können
- Code-Kommentare, die auf Features, Prozesse oder Systeme verweisen, die nicht mehr existieren
- Benutzerhandbücher oder Betriebsverfahren, die veraltete Systemschnittstellen beschreiben
- Historische Entscheidungsbegründung, die verloren ist, was unklar macht, warum Systeme so funktionieren, wie sie es tun

## Symptoms ▲

- [Verlängerte Recherchezeit](verlaengerte-recherchezeit.md)
<br/>  Entwickler verbringen übermäßig viel Zeit damit, Systemverständnis aus fragmentierten und veralteten Dokumentationsquellen zusammenzusetzen.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Teammitglieder haben Schwierigkeiten, produktiv zu werden, wenn Systemwissen nur in veralteten oder unzugänglichen Formaten existiert.
- [Schwierigkeit bei der Extraktion von Legacy-Geschäftslogik](schwierigkeit-bei-der-extraktion-von-legacy-geschaeftslogik.md)
<br/>  Wenn Dokumentation verloren oder veraltet ist, erfordert das Verständnis eingebetteter Geschäftslogik teure Code-Archäologie-Bemühungen.
- [Wissenslücken](wissensluecken.md)
<br/>  Kritische Systemwissenslücken entstehen, wenn Dokumentation veraltet und die Personen, die sie schrieben, ausgeschieden sind.
- [Scheiternde ROI-Rechtfertigung für Modernisierung](scheiternde-roi-rechtfertigung-fuer-modernisierung.md)
<br/>  Ohne klare Dokumentation von Systemfähigkeiten und -verhaltensweisen ist es unmöglich, Modernisierungsbemühungen akkurat einzugrenzen.

## Causes ▼

- [Informationsverfall](informationsverfall.md)
<br/>  Dokumentation, die einst akkurat war, verschlechtert sich über die Zeit, während sich das System weiterentwickelt, aber die Dokumentation nicht gepflegt wird.
- [Implizites Wissen](implizites-wissen.md)
<br/>  Kritisches Systemwissen wurde nie aufgeschrieben und existiert nur in den Köpfen von Entwicklern, die seitdem ausgeschieden sind.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Der Abgang erfahrener Entwickler nimmt unersetzliches Systemwissen mit sich und hinterlässt Lücken, die Dokumentation nicht füllen kann.
- [Unklare Verantwortlichkeit für Dokumentation](unklare-verantwortlichkeit-fuer-dokumentation.md)
<br/>  Ohne klare Verantwortung, Dokumentation aktuell zu halten, veraltet sie und wird schließlich obsolet.

## Detection Methods ○

- Audit bestehender Systemdokumentation auf Vollständigkeit, Genauigkeit und Zugänglichkeit
- Interview langjähriger Mitarbeiter zu Systemwissen und Identifikation von Wissenslücken
- Bewertung von Dokumentationsformaten und -werkzeugen auf Obsoleszenz- und Zugänglichkeitsprobleme
- Kartierung kritischen Systemwissens auf Einzelpersonen und Identifikation von Single Points of Failure
- Überprüfung von Codebasen auf undokumentierte Features oder Verhaltensweisen ohne Erklärung
- Testen des Teamverständnisses von Systemarchitektur und Geschäftsregeln durch Workshops
- Analyse der für Systemanalyse und Reverse-Engineering-Aktivitäten aufgewendeten Zeit
- Befragung von Entwicklungsteams zu Vertrauensniveaus beim Verständnis des Legacy-Systemverhaltens

## Examples

Ein Telekommunikationsunternehmen muss sein vor 15 Jahren gebautes Abrechnungssystem modernisieren. Die ursprüngliche Systemdokumentation existiert als Word-Dokumente auf Netzwerklaufwerken, die veraltete Software zum Öffnen erfordern, und die meisten Dateien sind beschädigt oder unvollständig. Der leitende Entwickler, der das System baute, verließ das Unternehmen vor 8 Jahren, und die zwei verbleibenden Teammitglieder, die daran arbeiteten, haben widersprüchliche Erinnerungen darüber, wie bestimmte Abrechnungsregeln funktionieren. Das Team entdeckt, dass das System Dutzende Sonderfälle für unterschiedliche Kundentypen, Werbeangebote und regulatorische Anforderungen handhabt, aber diese Regeln sind ohne Kommentare oder externe Dokumentation im Code eingebettet. Datenbank-Tabellennamen nutzen kryptische Abkürzungen, die für das ursprüngliche Team Sinn ergaben, aber jetzt bedeutungslos sind. Als sie versuchen zu verstehen, warum bestimmte Abrechnungsberechnungen spezifische Ergebnisse produzieren, müssen sie Tausende Zeilen unkommentierten Codes durchgehen, Datenbank-Trigger analysieren und Konfigurationsdateien untersuchen, die auf Geschäftsregeln verweisen, an deren Implementierung sich niemand erinnert. Die Dokumentations-Archäologie-Bemühung dauert 6 Monate und offenbart, dass das System mehrere Abrechnungspraktiken implementiert, die vom Geschäft nicht mehr genutzt werden, aber nicht sicher entfernt werden können, weil ihr Zweck und ihre Abhängigkeiten nicht verstanden werden.
