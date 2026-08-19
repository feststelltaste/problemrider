---
title: Datendeduplizierung
description: Erkennung und Beseitigung redundanter Daten in Speichersystemen.
category:
- Database
- Performance
problems:
- unbounded-data-growth
- code-duplication
- cross-system-data-synchronization-problems
- high-database-resource-utilization
- silent-data-corruption
- master-data-ownership-gaps
layout: solution
lang: de
en_slug: data-deduplication
related_solutions:
- slug: redundant-data-storage
  similarity: 0.8
- slug: data-integrity
  similarity: 0.8
- slug: compression
  similarity: 0.75
- slug: data-replication
  similarity: 0.75
- slug: data-archiving
  similarity: 0.75
- slug: data-strategy
  similarity: 0.75
---

## Description

Datendeduplizierung identifiziert Datensätze, Dateien oder Blöcke, die redundante Kopien derselben zugrunde liegenden Information sind, und konsolidiert sie in eine einzige maßgebliche Instanz, mittels exaktem Abgleich (Prüfsummen, Hashing) für identischen Inhalt oder unscharfem Abgleich (nach Namen, Adressen, Identifikatoren) für nahezu identische Datensätze, die durch inkonsistente Prozesse entstanden sind. In Legacy-Systemen häufen sich Duplikate typischerweise aus strukturellen Gründen an, nicht durch Zufall: mehrere Eingabekanäle, die dasselbe Kunden- oder Produktkonzept ohne gemeinsame Identitätsprüfung speisen, Migrationen, die bereits vorhandene Daten erneut importierten, oder das schlichte Fehlen von Eindeutigkeitsbeschränkungen am Speicherpunkt. Unbehandelt bläht diese Redundanz Speicher- und Verarbeitungskosten auf, aber wichtiger noch untergräbt sie das Vertrauen in die Daten selbst, da Berichte, Kundenzahlen und nachgelagerte Automatisierung alle still doppelt zählen oder dieselbe Entität mehrfach kontaktieren. Deduplizierung adressiert dies auf zwei Ebenen: ein einmaliger oder periodischer Bereinigungsdurchlauf, der bestehende Duplikate findet und zusammenführt, und präventive Beschränkungen oder Prüfungen, die verhindern, dass künftig neue Duplikate entstehen. Weil das Zusammenführen notwendigerweise die Entscheidung erfordert, welcher von mehreren widersprüchlichen Werten maßgeblich ist, ist Deduplizierung untrennbar mit der Etablierung eines Stammdaten-Eigentumsmodells verbunden, und es ist auch der Punkt, an dem Falsch-Positive am gefährlichsten sind, da eine falsche Zusammenführung still die Unterscheidbarkeit zweier Datensätze zerstört, die nie tatsächlich dieselbe Entität waren.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Prüfen Sie die Legacy-Datenbank auf doppelte Datensätze, indem Sie Schlüsselfelder analysieren und unscharfen Abgleich auf Namen, Adressen oder Identifikatoren anwenden
- Implementieren Sie Deduplizierung auf Speicherebene mittels inhaltsadressierbarem Speicher für Dateien und Dokumente
- Fügen Sie Eindeutigkeitsbeschränkungen und Deduplizierungsprüfungen auf Datenbankebene hinzu, um die Entstehung neuer Duplikate zu verhindern
- Gestalten Sie einen inkrementellen Deduplizierungsprozess, der neben Produktion laufen kann, ohne den Betrieb zu stören
- Etablieren Sie eine Stammdatenmanagement-Strategie, um maßgebliche Quellen für gemeinsam genutzte Daten zu definieren
- Nutzen Sie Prüfsummen oder Hashing, um doppelte Dateien in Dokumentenverwaltungssystemen zu erkennen
- Erstellen Sie Zusammenführungsstrategien für die Handhabung widersprüchlicher Attributwerte beim Konsolidieren doppelter Datensätze

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Reduziert Speicherkosten, indem redundante Kopien derselben Daten eliminiert werden
- Verbessert die Datenqualität, indem doppelte Datensätze zu einzelnen maßgeblichen Versionen konsolidiert werden
- Reduziert die Verarbeitungszeit für Operationen, die sonst über doppelte Daten iterieren
- Vereinfacht Data Governance durch eine einzige maßgebliche Quelle

**Kosten und Risiken:**
- Deduplizierungslogik kann fälschlicherweise unterschiedliche Datensätze zusammenführen, die ähnlich erscheinen (Falsch-Positive)
- Das Entfernen von Duplikaten aus Legacy-Systemen kann Anwendungen brechen, die von spezifischen doppelten Datensätzen abhängen
- Die anfängliche Deduplizierung großer Datensätze erfordert erhebliche Verarbeitungszeit und sorgfältige Validierung
- Die Pflege von Deduplizierungsregeln erfordert laufenden Aufwand, während sich Datenmuster weiterentwickeln

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-CRM-System häufte über ein Jahrzehnt über 2 Millionen Kundendatensätze an, wobei schätzungsweise 30 Prozent Duplikate waren, entstanden durch unterschiedliche Eingabekanäle (Telefon, Web, Filiale). Vertriebsmitarbeiter verschwendeten Zeit damit, denselben Kunden mehrfach zu kontaktieren, und Marketingkampagnen waren durch aufgeblähte Kundenzahlen verzerrt. Das Team implementierte eine Deduplizierungspipeline mittels unscharfem Abgleich auf Name, E-Mail und Telefonnummernfeldern, mit Konfidenzwerten, um wahrscheinliche Duplikate von unsicheren Übereinstimmungen zu unterscheiden. Duplikate mit hoher Konfidenz wurden automatisch zusammengeführt, während unsichere Fälle zur manuellen Überprüfung eingereiht wurden. Die Bereinigung reduzierte die Kundendatenbank um 28 Prozent, verbesserte die Kampagnen-Zielgenauigkeit und eliminierte Beschwerden von Kunden über doppelte Kontaktaufnahme.
