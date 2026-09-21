---
title: Kompatibilität als Fehlerkriterium
description: Behandlung von Kompatibilitätsregressionen als build-brechende Defekte,
  nicht als akzeptable technische Schulden.
category:
- Process
- Testing
problems:
- breaking-changes
- regression-bugs
- fear-of-breaking-changes
- quality-blind-spots
- quality-degradation
- insufficient-testing
layout: solution
lang: de
en_slug: compatibility-as-error
related_solutions:
- slug: compatibility-measurement
  similarity: 0.85
- slug: compatibility-governance
  similarity: 0.85
- slug: compatibility-standards
  similarity: 0.85
- slug: compatibility-testing
  similarity: 0.85
- slug: backward-compatibility
  similarity: 0.8
- slug: compatibility-certification
  similarity: 0.8
---

## Description

Kompatibilität als Fehlerkriterium ist die Praxis, jede abwärtsinkompatible Änderung an einer API, einem Schema oder einer Schnittstelle als build-brechenden Defekt zu behandeln, der ein Release blockiert, statt als akzeptablen Tradeoff oder ein Stück technischer Schulden, das später adressiert wird. Es wird durchgesetzt, indem automatisierte Kompatibilitätsprüfungen — Vertragstests, die eine vorgeschlagene Änderung gegen das Schema der vorherigen stabilen Version vergleichen — direkt in die CI-Pipeline eingebunden werden, sodass eine Regression abgefangen wird und das Mergen blockiert, bevor sie je Konsumenten erreicht, mit derselben Dringlichkeit, die normalerweise einem fehlgeschlagenen Sicherheitsscan vorbehalten ist. Diese Neuformulierung ist in Legacy-Kontexten wichtig, weil Kompatibilitätsbrüche dort tendenziell reaktiv behandelt werden: Eine brechende Änderung wird ausgeliefert, das System eines Integrationspartners fällt aus, und das Team hastet, den Schaden nachträglich zu flicken, wobei sich derselbe kostspielige Zyklus Release für Release wiederholt. Kompatibilitätsfehler durch Richtlinie release-blockierend zu machen verwandelt diese reaktive Haltung in eine proaktive, da die Kosten eines Bruchs nun sofort vom Autor der Änderung getragen werden, in Form eines fehlgeschlagenen Builds, statt nachgelagert von jedem Integrationskonsumenten Wochen oder Monate später. Es verbietet bewusste Breaking Changes nicht vollständig, aber leitet sie durch ein explizites Genehmigungsgate, das einen formulierten Migrationsplan verlangt, was bewusste, koordinierte Evolution von versehentlicher Regression unterscheidet. Das offensichtliche Risiko ist, dass übermäßig strenge oder schlecht abgestimmte Prüfungen False Positives erzeugen, die das Vertrauen in das Gate erodieren und Teams einladen, es zu umgehen, sodass die Kompatibilitäts-Test-Suite selbst vertrauenswürdig genug sein muss, um zu rechtfertigen, ein Release auf ihrem Ergebnis zu blockieren.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Fügen Sie der CI-Pipeline Kompatibilitätsprüfungen hinzu, die den Build bei jeder abwärtsinkompatiblen Änderung scheitern lassen
- Nutzen Sie Vertragstest-Werkzeuge, um API- oder Schema-Regressionen automatisch zu erkennen
- Definieren Sie Kompatibilität als release-blockierendes Kriterium in Ihrer Definition of Done
- Erstellen Sie automatisierte Kompatibilitäts-Test-Suiten, die gegen die vorherige stabile Version laufen
- Behandeln Sie Kompatibilitätsfehler mit derselben Dringlichkeit wie Sicherheitslücken: beheben vor dem Mergen
- Etablieren Sie ein Review-Gate, bei dem bewusste Breaking Changes explizite Genehmigung und einen Migrationsplan erfordern

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Fängt Breaking Changes ab, bevor sie Konsumenten erreichen, und verhindert Produktionsvorfälle
- Verschiebt die Teamdenkweise von reaktiven Kompatibilitätsfixes zu proaktiver Kompatibilitätssicherung
- Verringert die Gesamtkosten von Integrationsfehlern über die Organisation hinweg

**Kosten und Risiken:**
- Kann die Entwicklung verlangsamen, wenn die Pipeline bei Kompatibilitätsprüfungen blockiert
- Erfordert Investition in Tooling und Testinfrastruktur für Kompatibilitätsvalidierung
- Übermäßig strenge Regeln könnten Teams frustrieren, die bewusste Breaking Changes vornehmen müssen
- False Positives in Kompatibilitätsprüfungen können das Vertrauen in den Prozess erodieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Zahlungsabwicklungsunternehmen erlebte vierteljährliche Vorfälle, bei denen API-Änderungen Händlerintegrationen brachen. Das Team fügte seiner CI-Pipeline einen Vertragstest-Schritt hinzu, der jeden Pull Request gegen das aktuell deployte API-Schema verglich. Jede inkompatible Änderung ließ den Build sofort scheitern. Im ersten Jahr sank die Anzahl kompatibilitätsbezogener Produktionsvorfälle von zwölf auf einen, und dieser eine Vorfall wurde auf einen Konfigurationsfehler statt eine Codeänderung zurückgeführt.
