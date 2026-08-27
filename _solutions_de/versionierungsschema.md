---
title: Versionierungsschema
description: Definition, wann und warum sich Versionsnummern ändern, um
  Kompatibilitätsabsicht zu signalisieren.
category:
- Process
- Dependencies
problems:
- api-versioning-conflicts
- breaking-changes
- legacy-api-versioning-nightmare
- dependency-version-conflicts
- integration-difficulties
- change-management-chaos
layout: solution
lang: de
en_slug: versioning-scheme
related_solutions:
- slug: semantic-versioning
  similarity: 0.85
- slug: version-control
  similarity: 0.75
- slug: api-versioning-strategy
  similarity: 0.75
- slug: compatibility-standards
  similarity: 0.7
- slug: schema-registry
  similarity: 0.7
- slug: compatibility-governance
  similarity: 0.7
---

## Description

Ein Versionierungsschema ist eine explizite, dokumentierte Richtlinie dafür, wann und warum sich eine Versionsnummer ändert, die präzise definiert, was einen Breaking Change von einer Feature-Ergänzung von einem Bugfix für ein gegebenes Artefakt unterscheidet — eine Bibliothek, ein Datenformat oder eine API —, sodass die Versionsnummer selbst echte Information kommuniziert, statt ein willkürlicher Zähler zu sein. Semantic Versioning, datumsbasierte Versionierung und URI-basierte Versionierung passen jeweils zu verschiedenen Arten von Artefakten, und die Wahl zählt weniger als die Konsistenz und Klarheit, mit der sie angewendet und dokumentiert wird. Legacy-Umgebungen sammeln häufig Komponenten an, die nie überhaupt sinnvoll versioniert wurden — Build-Nummern werden erhöht, ohne definierte Beziehung zu Kompatibilität —, was nachgelagerten Konsumenten kein zuverlässiges Signal darüber gibt, ob ein Update sicher übernommen werden kann oder sorgfältiges Review erfordert, was wiederum manuelle, fallweise Untersuchung vor jedem Upgrade erzwingt. Die Einführung eines bewussten Versionierungsschemas, selbst rückwirkend durch Prüfung des aktuellen Zustands einer Legacy-Komponente und Zuweisung einer sinnvollen Basisversion, stellt dieses Signal wieder her und macht es möglich, Upgrade-Entscheidungen mit Zuversicht zu automatisieren, da eine Versionserhöhung jetzt zuverlässig die Art der Änderung anzeigt. Dies ist eine vergleichsweise kostengünstige Intervention relativ zu ihrer Wirkung: Sie erfordert laufende Disziplin von den Teams, die Versionserhöhungen anwenden, aber sie beseitigt einen erheblichen manuellen Koordinations-Overhead über ein Legacy-Portfolio hinweg, besonders eines, das aus vielen voneinander abhängigen internen Bibliotheken und Diensten besteht.

## How to Apply ◆

- Wählen Sie ein zum Artefakttyp passendes Versionierungsschema: Semantic Versioning für Bibliotheken, datumsbasierte Versionierung für Datenformate, oder URI-basierte Versionierung für APIs.
- Dokumentieren Sie die Versionierungsrichtlinie explizit und definieren Sie, was einen Breaking Change, eine Feature-Ergänzung und einen Bugfix im Kontext Ihres Legacy-Systems ausmacht.
- Wenden Sie das Versionierungsschema rückwirkend auf Legacy-Komponenten an, indem Sie ihren aktuellen Zustand prüfen und eine Basisversion zuweisen.
- Integrieren Sie Versionsvalidierung in Build- und Deployment-Pipelines, um das Schema durchzusetzen.
- Kommunizieren Sie Versionsänderungen durch Changelogs, Release Notes und automatisierte Benachrichtigungen an nachgelagerte Konsumenten.
- Überprüfen Sie das Versionierungsschema periodisch, um sicherzustellen, dass es der sich entwickelnden Systemlandschaft weiterhin dient.

## Tradeoffs ⇄

**Vorteile:**
- Gibt Konsumenten ein zuverlässiges Signal über die Art und das Risiko eines Updates.
- Ermöglicht Automatisierung von Abhängigkeitsverwaltung und Upgrade-Entscheidungen.
- Schafft eine gemeinsame Sprache für die Diskussion von Änderungsauswirkungen über Teams hinweg.

**Kosten:**
- Erfordert Teamdisziplin, um Änderungen korrekt zu kategorisieren und Versionen entsprechend zu erhöhen.
- Die Wahl des falschen Schemas kann Verwirrung statt Klarheit schaffen.
- Die Nachrüstung von Versionen auf Legacy-Komponenten ohne vorherige Versionierung erfordert sorgfältige Analyse.
- Unterschiedliche Versionierungsschemata über das Portfolio hinweg können die beabsichtigte Klarheit reduzieren.

## How It Could Be

Ein großes Unternehmen pflegt über fünfzig interne Bibliotheken, die von Legacy-Anwendungen genutzt werden. Historisch wurden Bibliotheken mit willkürlichen Build-Nummern versioniert, die keine Kompatibilitätsinformation vermittelten. Das Architekturteam führt ein einheitliches Versionierungsschema ein: Semver für Bibliotheken, kalenderbasierte Versionierung für Datenschemata, und URL-Pfad-Versionierung für REST-APIs. Jedes Team dokumentiert seine Versionierungsrichtlinie in einem gemeinsam genutzten Wiki. Build-Pipelines setzen durch, dass Pull Requests eine Versionserhöhung und einen Changelog-Eintrag enthalten. Innerhalb weniger Monate können Entwickler in der gesamten Organisation Upgrade-Risiko auf einen Blick einschätzen, und automatisierte Werkzeuge handhaben Patch-Ebenen-Updates ohne menschliches Eingreifen.
