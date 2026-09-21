---
title: Failover-Cluster
description: Redundante Pflege von Servern oder Systemen als funktionale Gruppe.
category:
- Architecture
- Operations
problems:
- single-points-of-failure
- system-outages
- cascade-failures
- slow-incident-resolution
- capacity-mismatch
- high-maintenance-costs
- deployment-risk
layout: solution
lang: de
en_slug: failover-cluster
related_solutions:
- slug: failover-mechanisms
  similarity: 0.85
- slug: redundancy
  similarity: 0.8
- slug: high-availability-architectures
  similarity: 0.8
- slug: load-balancing
  similarity: 0.75
- slug: data-replication
  similarity: 0.75
- slug: retry
  similarity: 0.75
---

## Description

Ein Failover-Cluster hält zwei oder mehr Server als koordinierte Gruppe laufend — typischerweise Active-Passive oder Active-Active — mit gemeinsamem oder repliziertem Speicher, sodass beim Erkennen eines Ausfalls des aktiven Knotens durch eine Gesundheitsprüfung der Verkehr automatisch zu einem Standby-Knoten umgeleitet wird, der bereits Zugriff auf den aktuellen Zustand hat. Dies adressiert direkt das Single-Point-of-Failure-Problem, das sich in Legacy-Systemen ansammelt, wenn ein kritischer Dienst ursprünglich auf einem einzigen Server bereitgestellt wurde, weil Clustering nie geplant war, was jeden Hardwareausfall oder jedes OS-Patch zu einem geplanten Ausfall statt zu einer routinemäßigen, unsichtbaren Wartungsaktivität macht. Die Einführung von Clustering zuerst für die geschäftskritischsten Dienste, mit automatischen Gesundheitsprüfungen und getesteten Failover-Auslösern, verwandelt frühere vollständige Dienstunterbrechungen in kurze automatische Übergänge und gibt Betriebsteams die Fähigkeit, durch bewusstes Failover zu patchen oder andere Wartung durchzuführen, statt den gesamten Dienst abzuschalten. Der Zielkonflikt sind die laufenden Kosten und die betriebliche Komplexität des Betriebs redundanter Infrastruktur — Quorum-Regeln, Split-Brain-Vermeidung und Änderungen an der Sitzungsverteilung, die die Legacy-Anwendung selbst möglicherweise braucht —, plus die Disziplin, Failover ausreichend regelmäßig zu testen, damit es bei einem echten Ausfall tatsächlich funktioniert und nicht nur in der Theorie.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Inventarisieren Sie alle Single-Point-of-Failure-Komponenten im Legacy-System und priorisieren Sie sie nach geschäftlicher Kritikalität
- Führen Sie Active-Passive- oder Active-Active-Clustering zuerst für die kritischsten Dienste ein
- Konfigurieren Sie gemeinsamen Speicher oder replizierte Datenspeicher, damit Failover-Knoten Zugriff auf den aktuellen Zustand haben
- Richten Sie automatische Gesundheitsprüfungen und Failover-Auslöser mit angemessenen Timeout-Schwellenwerten ein
- Testen Sie Failover-Szenarien regelmäßig in Staging-Umgebungen, die die Produktionstopologie widerspiegeln
- Dokumentieren Sie den Failover-Prozess in Runbooks, damit Bereitschaftspersonal eingreifen kann, wenn automatisches Failover nicht anspringt
- Erweitern Sie Clustering schrittweise auf sekundäre Dienste, sobald das Team betriebliches Vertrauen gewinnt

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt Single Points of Failure für kritische Legacy-Dienste
- Verringert ungeplante Ausfallzeit und deren geschäftliche Auswirkung
- Ermöglicht Wartungsfenster ohne vollständige Dienstunterbrechung
- Bietet eine Grundlage für künftige Hochverfügbarkeitsverbesserungen

**Kosten und Risiken:**
- Erhöhte Infrastrukturkosten für redundante Hardware oder Cloud-Instanzen
- Betriebliche Komplexität wächst mit Cluster-Management, Quorum-Regeln und Split-Brain-Vermeidung
- Legacy-Anwendungen benötigen möglicherweise Änderungen zur Unterstützung von Sitzungsverteilung oder zustandslosem Betrieb
- Failover-Tests erfordern sorgfältige Planung, um versehentliche Produktionsausfälle zu vermeiden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Einzelhandelsunternehmen betrieb sein Auftragsverarbeitungssystem auf einem einzelnen Legacy-Anwendungsserver. Jeder Hardwareausfall oder jedes OS-Patch erforderte ein vollständiges Wartungsfenster, was Stunden an Umsatzverlust kostete. Durch die Einführung eines zweiknotigen Active-Passive-Failover-Clusters mit gemeinsamem Datenbankspeicher reduzierte das Team ungeplante Ausfallzeit um über 90 %. Der passive Knoten übernahm den Verkehr automatisch innerhalb von Sekunden nach Erkennen eines Heartbeat-Verlusts, und geplante Wartung konnte fortgesetzt werden, indem vor dem Anwenden von Patches sanft auf Failover umgeschaltet wurde.
