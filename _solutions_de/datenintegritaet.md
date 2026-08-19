---
title: Datenintegrität
description: Mechanismen zur Sicherstellung von Datengenauigkeit, -konsistenz und
  -zuverlässigkeit.
category:
- Database
problems:
- silent-data-corruption
- data-migration-integrity-issues
- cross-system-data-synchronization-problems
- database-schema-design-problems
- inconsistent-behavior
- unbounded-data-growth
- cache-invalidation-problems
- dma-coherency-issues
- synchronization-problems
layout: solution
lang: de
en_slug: data-integrity
related_solutions:
- slug: continuous-data-verification
  similarity: 0.8
- slug: checksums
  similarity: 0.8
- slug: data-deduplication
  similarity: 0.8
- slug: fault-tolerant-data-structures
  similarity: 0.75
- slug: error-correction-codes
  similarity: 0.75
- slug: monitoring-system-integrity
  similarity: 0.75
---

## Description

Datenintegrität umfasst die Beschränkungen und Mechanismen — Fremdschlüssel, Eindeutigkeits- und Prüfbeschränkungen, transaktionale Atomarität und ergänzende Validierung auf Anwendungsebene —, die gespeicherte Daten genau, intern konsistent und frei von Widersprüchen wie verwaisten Referenzen oder unmöglichen Werten halten. Der Kernmechanismus ist Verteidigung auf mehreren Ebenen: Beschränkungen auf Datenbankebene wirken als harte Absicherung, die ungültige Zustände ablehnt, unabhängig davon, welcher Anwendungscodepfad versuchte, sie zu erzeugen, während Validierung auf Anwendungsebene Probleme früher erkennt und bessere Fehlermeldungen liefert, und keine ersetzt die andere. In Legacy-Systemen fehlen Integritätsbeschränkungen häufig vollständig, weil sie während der ursprünglichen Entwicklung nie hinzugefügt oder bewusst gelockert wurden, um ein inzwischen vergessenes Hindernis zu umgehen, und das Ergebnis nach Jahren des Betriebs ist eine Anhäufung still korrumpierter Daten: Kontakte, die auf gelöschte Unternehmen verweisen, doppelte Entitäten und Inkonsistenzen, die nur auftauchen, wenn jemand versucht, einen zuverlässigen Bericht oder eine Migration auf den Daten aufzubauen. Die Integrität eines solchen Systems wiederherzustellen ist notwendigerweise schrittweise, da Beschränkungen nicht einfach über Daten aktiviert werden können, die sie bereits verletzen — bestehende Verstöße müssen zuerst gefunden und behoben werden, oft mittels zweckgebauter Bereinigungsskripte, bevor die Beschränkung, die sie verhindert hätte, sicher aktiviert werden kann. Einmal etabliert, verwandeln diese Beschränkungen Datenqualitätsprobleme von einer wiederkehrenden Untersuchungslast in Build-Zeit- oder Transaktionszeit-Fehler, die sofort auftauchen, in dem Moment, in dem der schlechte Zustand sonst erzeugt worden wäre.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Prüfen Sie bestehende Datenbankschemata auf fehlende Beschränkungen (Fremdschlüssel, Eindeutigkeitsbeschränkungen, Prüfbeschränkungen, Nicht-Null)
- Fügen Sie Beschränkungen auf Datenbankebene schrittweise hinzu, beginnend mit den kritischsten Geschäftsentitäten
- Implementieren Sie Validierung auf Anwendungsebene als Ergänzung zu Datenbankbeschränkungen, nicht als Ersatz
- Nutzen Sie Transaktionen angemessen, um Atomarität mehrstufiger Datenoperationen sicherzustellen
- Fügen Sie referenzielle Integritätsbeschränkungen zwischen verwandten Tabellen hinzu, die im ursprünglichen Design möglicherweise ausgelassen wurden
- Implementieren Sie Datenqualitätsüberwachung, die kontinuierlich auf verwaiste Datensätze, Duplikate und Beschränkungsverletzungen prüft
- Erstellen Sie Datenreparaturskripte für bekannte Integritätsprobleme und führen Sie sie als Teil regelmäßiger Wartung aus

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verhindert, dass korrupte oder inkonsistente Daten auf Datenbankebene ins System gelangen
- Reduziert den Bedarf an teuren Datenbereinigungs- und Abgleichprozessen
- Erhöht das Vertrauen in Daten für Berichte, Analytics und nachgelagerte Integrationen
- Macht implizite Datenregeln explizit und durchsetzbar

**Kosten und Risiken:**
- Das Hinzufügen von Beschränkungen zu Legacy-Datenbanken mit bestehenden schlechten Daten erfordert zunächst Datenbereinigung
- Strenge Beschränkungen können Legacy-Code brechen, der sich auf lockere Validierung verließ
- Fremdschlüsselbeschränkungen können die Schreibperformance bei hochdurchsatzigen Tabellen beeinträchtigen
- Die rückwirkende Durchsetzung von Integrität auf historische Daten kann extrem zeitaufwendig sein

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-CRM-System hatte keine Fremdschlüsselbeschränkungen in seiner Datenbank. Über Jahre des Betriebs häuften sich verwaiste Datensätze an: Kontakte verwiesen auf gelöschte Unternehmen, Aktivitäten waren mit nicht existierenden Opportunities verknüpft, und doppelte Datensätze vermehrten sich. Das Team begann damit, die Daten zu profilieren, um Integritätsverletzungen zu quantifizieren, und fand über 50.000 verwaiste Datensätze über 12 Tabellen. Sie schrieben Bereinigungsskripte, um bestehende Verstöße zu beheben, und fügten dann Fremdschlüsselbeschränkungen mit für jede Beziehung angemessenen Kaskadierungsregeln hinzu. Anwendungscode, der still verwaiste Datensätze erzeugt hatte, begann Fehler zu werfen, die einer nach dem anderen behoben wurden. Nach sechs Monaten sanken die von Vertriebsmitarbeitern gemeldeten Datenqualitätsprobleme von wöchentlichen Vorkommnissen auf nahezu null.
