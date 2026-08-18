---
title: Fehlschläge des Strangler-Fig-Patterns
description: Inkrementelle Modernisierung mithilfe des Strangler-Fig-Patterns stockt
  aufgrund komplexer gegenseitiger Abhängigkeiten und Herausforderungen der Datenkonsistenz.
category:
- Architecture
- Code
- Operations
related_problems:
- slug: integration-difficulties
  similarity: 0.65
- slug: modernization-strategy-paralysis
  similarity: 0.6
- slug: modernization-roi-justification-failure
  similarity: 0.6
- slug: complex-implementation-paths
  similarity: 0.6
- slug: architectural-mismatch
  similarity: 0.6
- slug: stagnant-architecture
  similarity: 0.6
solutions:
- strangler-fig-pattern
- walking-skeleton
- mikado-method
- anti-corruption-layer
- feature-toggles
- characterization-tests
- bubble-context
- incremental-refactoring
- parallel-run
- large-scale-refactoring
layout: problem
lang: de
en_slug: strangler-fig-pattern-failures
---

## Description

Fehlschläge des Strangler-Fig-Patterns treten auf, wenn Versuche, Legacy-Systemkomponenten graduell durch moderne Alternativen zu ersetzen, aufgrund unterschätzter Komplexität in Systemgrenzen, Anforderungen an Datenkonsistenz und gegenseitigen Abhängigkeiten stocken oder scheitern. Das Strangler-Fig-Pattern, das dazu gedacht ist, risikoarme inkrementelle Modernisierung zu ermöglichen, wird zu einer Quelle erhöhter Komplexität und technischer Schulden, wenn der „Erwürgungs"-Prozess nicht abgeschlossen werden kann, was Organisationen mit Hybridsystemen zurücklässt, die komplexer sind, als es entweder das ursprüngliche Legacy-System oder ein vollständiger Ersatz gewesen wären.

## Indicators ⟡

- Strangler-Fig-Implementierungsprojekte, die konsequent Termine und Meilensteine verpassen
- Schwierigkeiten, saubere Grenzen zwischen Legacy- und neuen Systemkomponenten zu identifizieren
- Komplexität der Datensynchronisation zwischen Legacy- und neuen Komponenten, die Erwartungen übersteigt
- Neue Systemkomponenten, die zunehmend komplexe Integration mit verbleibenden Legacy-Teilen erfordern
- Performance-Verschlechterung, während Anfragen durch sowohl Legacy- als auch neue Systemschichten fließen
- Team-Schätzungen zur Fertigstellung des „Erwürgungs"-Prozesses, die sich stetig verlängern
- Wachsende operative Komplexität durch die gleichzeitige Verwaltung von Legacy- und neuen Systemkomponenten

## Symptoms ▲

- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Die ins Stocken geratene Strangler-Fig-Migration verursacht, dass das Modernisierungsprojekt wiederholt Termine verpasst, während die Komplexität eskaliert.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Die gleichzeitige Verwaltung von Legacy- und neuen Komponenten verdoppelt operativen Overhead und Wartungsaufwand.
- [Systemausfälle](systemausfaelle.md)
<br/>  Fehler bei der Datensynchronisation und Performance-Probleme im Hybridsystem verursachen Serviceunterbrechungen.
- [Budgetüberschreitungen](budgetueberschreitungen.md)
<br/>  Die unerwartete Komplexität der Fertigstellung der Migration verursacht, dass Kosten die ursprünglichen Schätzungen erheblich übersteigen.
- [Vertrauensverlust bei Stakeholdern](vertrauensverlust-bei-stakeholdern.md)
<br/>  Wiederholte Verzögerungen und eskalierende Kosten in der Modernisierungsbemühung untergraben das Vertrauen der Stakeholder in den technischen Ansatz.

## Causes ▼

- [Versteckte Abhängigkeiten](versteckte-abhaengigkeiten.md)
<br/>  Unentdeckte Abhängigkeiten zwischen Legacy-Komponenten machen es unmöglich, einzelne Teile sauber zu trennen und zu ersetzen.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Eng gekoppelte Legacy-Komponenten widersetzen sich der sauberen Grenzidentifikation, die für inkrementellen Ersatz nötig ist.
- [Probleme bei der systemübergreifenden Datensynchronisation](probleme-bei-der-systemuebergreifenden-datensynchronisation.md)
<br/>  Herausforderungen der Datenkonsistenz zwischen Legacy- und modernen Komponenten untergraben den inkrementellen Migrationsansatz.
- [Komplexes Domänenmodell](komplexes-domaenenmodell.md)
<br/>  Inhärent komplexe Geschäftsdomänen machen es schwierig, saubere Grenzen für inkrementellen Ersatz zu identifizieren.

## Detection Methods ○

- Verfolgung von Fortschrittsmetriken für die Strangler-Fig-Implementierung gegen ursprüngliche Zeitplanschätzungen
- Überwachung von Datenkonsistenzproblemen und Synchronisationsfehlern zwischen Systemkomponenten
- Messung von Systemkomplexitätsmetriken vor und während des Erwürgungsprozesses
- Bewertung von Team-Vertrauensniveaus und Schätzgenauigkeit für die Fertigstellung verbleibender Modernisierungsarbeit
- Analyse von Performance-Auswirkungen und operativem Overhead des Hybrid-Systemzustands
- Überprüfung der Anhäufung technischer Schulden in Integrations- und Synchronisationscode
- Befragung von Entwicklungsteams zu Herausforderungen und Blockaden bei der Fortsetzung der Modernisierung
- Bewertung, ob das aktuelle Hybridsystem besseren Wert bietet als das ursprüngliche Legacy-System

## Examples

Ein Einzelhandelsunternehmen beginnt, sein Bestandsverwaltungssystem mithilfe des Strangler-Fig-Patterns zu modernisieren, beginnend mit der Produktkatalogkomponente. Der neue Katalogservice funktioniert anfangs gut, aber als sie versuchen, die Preis-Engine zu ersetzen, entdecken sie, dass die Preislogik tief mit Bestandszuweisung, Auftragsverarbeitung und Promotionssystemen verwoben ist. Die Aufrechterhaltung der Datenkonsistenz zwischen dem neuen Katalog, dem Legacy-Preissystem und verschiedenen nachgelagerten Systemen erfordert komplexe Echtzeitsynchronisation, die während Spitzenzeiten häufig fehlschlägt. Jeder zusätzliche Komponentenersatz offenbart neue Abhängigkeiten, die in der ursprünglichen Systemanalyse nicht erkennbar waren. Nach 18 Monaten hat das Team 40 % des Legacy-Systems ersetzt, schätzt aber, dass die Fertigstellung der Modernisierung aufgrund zunehmender Komplexität weitere 3 Jahre dauern wird. Das Hybridsystem erfordert jetzt mehr operativen Overhead als das ursprüngliche Legacy-System, funktioniert bei Spitzenlasten schlechter und hat Datenkonsistenzfehler eingeführt, die zuvor nicht existierten. Die Organisation steht vor der schwierigen Wahl, die Modernisierungsbemühung aufzugeben oder sich zu weiteren Jahren an Arbeit mit unsicheren Ergebnissen zu verpflichten.
