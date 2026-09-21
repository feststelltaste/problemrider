---
title: Codegenerierung
description: Automatische Erstellung von Code-Teilen basierend auf Templates oder
  Metadaten.
category:
- Code
- Process
problems:
- code-duplication
- copy-paste-programming
- inconsistent-codebase
- maintenance-overhead
- slow-feature-development
- increased-cost-of-development
- increased-risk-of-bugs
layout: solution
lang: de
en_slug: code-generation
related_solutions:
- slug: automated-code-migration
  similarity: 0.7
- slug: static-analysis-and-linting
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
- slug: development-workflow-automation
  similarity: 0.7
- slug: ci-cd-pipeline
  similarity: 0.65
- slug: code-review-process-reform
  similarity: 0.65
---

## Description

Codegenerierung ist die automatisierte Produktion von Quellcode aus einer Vorlage, einem Schema oder einer anderen maschinenlesbaren Spezifikation, was manuell geschriebenen, repetitiven Boilerplate durch mechanisch und konsistent aus einer einzigen Quelle der Wahrheit abgeleitete Ausgabe ersetzt. Typische Kandidaten sind Data-Access-Objekte, API-Client-Stubs, Serialisierungslogik und Konfigurationsklassen — strukturell vorhersehbarer Code, der nur in Details variiert, die bereits anderswo erfasst sind, wie ein Datenbankschema oder eine API-Spezifikation. In Legacy-Modernisierungsarbeit ist dies besonders wertvoll, wenn viele strukturell ähnliche Komponenten auf einmal migriert werden, zum Beispiel wenn eine Repository-Klasse, ein DTO und ein Endpunkt für jede von Hunderten Legacy-Datenbanktabellen generiert werden, statt jede von Hand zu schreiben, was sowohl die Migration beschleunigt als auch garantiert, dass Namensgebung, Fehlerbehandlung und Abbildungskonventionen überall identisch angewendet werden, statt subtil von einer handgeschriebenen Datei zur nächsten abzudriften. Weil der generierte Code aus seinen Quellmetadaten abgeleitet wird statt direkt verfasst zu werden, verbreitet die Aktualisierung der Vorlage und Neugenerierung eine Änderung einheitlich über jedes generierte Artefakt in einem Schritt, was sonst eine mühsame und fehleranfällige Aktivität ist, von Hand über einen großen Legacy-Bestand durchzuführen. Dieser Vorteil kommt mit einer entsprechenden Abhängigkeit: Die Generierungsvorlagen und das Tooling selbst werden zu kritischer Infrastruktur, die gepflegt werden muss, und Entwickler müssen generierte Ausgabe gut genug verstehen, um sie zu debuggen, wenn etwas schiefgeht, was schwieriger sein kann als das Debuggen von Code, den sie selbst geschrieben haben. Generierten Code klar von handgeschriebenem Code getrennt zu halten und den Generierungsschritt in den Build einzubinden, sodass er nicht still aus der Synchronisation mit seiner Quelle geraten kann, ist es, was diesen Ansatz über die Zeit wartbar hält.

## How to Apply ◆

> In Legacy-Systemen verringert Codegenerierung Boilerplate-Duplizierung und setzt Konsistenz durch, indem repetitiver Code aus Vorlagen oder Metadaten generiert wird, statt von Hand geschrieben zu werden.

- Identifizieren Sie repetitive Muster in der Legacy-Codebasis, die einer vorhersehbaren Struktur folgen — Data-Access-Objekte, API-Client-Stubs, Serialisierungscode und Konfigurationsklassen sind übliche Kandidaten.
- Wählen Sie für den Technologie-Stack des Legacy-Systems angemessene Generierungswerkzeuge (Codegeneratoren, Template-Engines, Annotation-Prozessoren oder schemagetriebene Generatoren wie OpenAPI oder Protocol Buffers).
- Generieren Sie Code aus einer einzigen Quelle der Wahrheit (Datenbankschemata, API-Spezifikationen oder Konfigurationsdateien), um Konsistenz über die generierten Artefakte hinweg sicherzustellen.
- Halten Sie generierten Code klar von handgeschriebenem Code getrennt, durch Namenskonventionen, Verzeichnisstruktur oder Build-Werkzeug-Konfiguration, sodass Entwickler generierte Dateien nicht versehentlich ändern.
- Beziehen Sie den Generierungsschritt in die Build-Pipeline ein, sodass generierter Code mit seinen Quellmetadaten synchron bleibt.
- Nutzen Sie Codegenerierung während der Legacy-Migration, um konsistentes Boilerplate für das neue System basierend auf Legacy-Schema- oder Schnittstellendefinitionen zu produzieren.

## Tradeoffs ⇄

> Codegenerierung eliminiert Boilerplate-Wartung, führt aber Abhängigkeiten von Generierungswerkzeugen und Vorlagen ein, die verwaltet werden müssen.

**Vorteile:**

- Eliminiert ganze Klassen von Copy-Paste-Bugs, indem repetitiver Code konsistent aus einer einzigen Vorlage generiert wird.
- Beschleunigt die Entwicklung repetitiver Codestrukturen, besonders bei der Migration vieler ähnlicher Komponenten aus einem Legacy-System.
- Stellt Konsistenz über generierte Artefakte hinweg sicher — wenn sich die Vorlage ändert, ändert sich der gesamte generierte Code einheitlich.
- Verringert die Menge an Code, die Entwickler schreiben und überprüfen müssen, was ihre Aufmerksamkeit auf Geschäftslogik fokussiert.

**Kosten und Risiken:**

- Generierter Code kann schwierig zu debuggen sein, wenn Probleme im Generierungsprozess statt in der generierten Ausgabe auftreten.
- Die Generierungsvorlagen und das Tooling werden zu kritischen Abhängigkeiten, die Wartung und Expertise erfordern.
- Übermäßiges Vertrauen in Codegenerierung kann zu generiertem Code führen, der nicht in allen Kontexten gut passt, was Workarounds erfordert.
- Entwickler könnten den generierten Code nicht gut genug verstehen, um Probleme zu debuggen oder zu erkennen, wann Generierung suboptimale Ausgabe produziert.

## How It Could Be

> Das folgende Szenario zeigt, wie Codegenerierung die Legacy-Systemmigration beschleunigt.

Ein Finanzdienstleistungsunternehmen migrierte von einem Legacy-System mit 180 Datenbanktabellen zu einer neuen Microservices-Architektur. Jede Tabelle brauchte eine entsprechende Repository-Klasse, ein DTO, einen Mapper und einen REST-Endpunkt im neuen System — ungefähr 900 Boilerplate-Dateien. Statt diese von Hand zu schreiben, baute das Team einen Codegenerator, der das Legacy-Datenbankschema las und alle vier Artefakte für jede Tabelle produzierte. Der Generator schloss in Sekunden ab, was Wochen manueller Codierung gebraucht hätte, und stellte sicher, dass Namenskonventionen, Fehlerbehandlungsmuster und Abbildungslogik über alle 180 Entitäten hinweg perfekt konsistent waren. Als das Team später beschloss, das Fehlerantwortformat über alle Endpunkte hinweg zu ändern, aktualisierten sie die Vorlage und regenerierten alle Endpunktklassen in einem einzigen Schritt.
