---
title: Threat Intelligence
description: Sammlung und Analyse von Informationen über aktuelle
  Bedrohungen und Angriffsmethoden.
category:
- Security
problems:
- monitoring-gaps
- knowledge-gaps
- obsolete-technologies
- regulatory-compliance-drift
- quality-blind-spots
- slow-incident-resolution
layout: solution
lang: de
en_slug: threat-intelligence
related_solutions:
- slug: honeypots
  similarity: 0.8
- slug: threat-modeling
  similarity: 0.8
- slug: security-monitoring
  similarity: 0.8
- slug: endpoint-detection-and-response
  similarity: 0.75
- slug: vulnerability-scans
  similarity: 0.75
- slug: security-metrics
  similarity: 0.75
---

## Description

Threat Intelligence ist die systematische Sammlung, Korrelation und Interpretation von Informationen über aktive Angreifer, ihre Werkzeuge und ihre Methoden, gesammelt aus Schwachstellendatenbanken, Anbieter-Advisories, Branchen-Sharing-Communities und dedizierten Feeds. Statt auf einen Vorfall zu warten, der offenbart, dass ein System exponiert ist, nutzen Teams diese externe Information, um vorherzusehen, welche Bedrohungen derzeit in freier Wildbahn ausgenutzt werden, und zu prüfen, ob ihr eigener Technologie-Stack ein plausibles Ziel ist. Für Legacy-Systeme trägt diese Praxis besonderes Gewicht: Die beteiligten Plattformen, Protokolle und Bibliotheken sind oft alt genug, dass sich die Mainstream-Sicherheitsaufmerksamkeit anderswohin verlagert hat, sodass die wenigen aktiven Offenlegungen, die für sie tatsächlich auftauchen, eher hochrelevant und zeitkritisch sind als Hintergrundrauschen. Da Legacy-Umgebungen häufig moderne Instrumentierung vermissen und sich nicht auf die Patch-Taktung von Anbietern verlassen können, die neuere Stacks genießen, wird externe Threat Intelligence zu einem der wenigen Frühwarnmechanismen, die verfügbar sind, bevor eine ausgenutzte Schwäche zu einem Vorfall wird. Es hilft auch, ein abstraktes Inventar alter Software in eine priorisierte Liste konkreter, derzeit aktiver Risiken zu übersetzen, was essenziell ist, wenn Patching-Kapazität begrenzt ist und jede Härtungsanstrengung gegen konkurrierende Wartungsanforderungen gerechtfertigt werden muss. Gut genutzt verschiebt es Sicherheitsarbeit von reaktivem Feuerwehrlöschen hin zu informierter Antizipation, verankert in dem, was Angreifer gerade tatsächlich tun.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Abonnieren Sie Threat-Intelligence-Feeds, die für den Technologie-Stack und die Branche des Legacy-Systems relevant sind
- Überwachen Sie Schwachstellendatenbanken (CVE, NVD) auf Offenlegungen, die Legacy-Komponenten und -Abhängigkeiten betreffen
- Nehmen Sie an branchenspezifischen Informationsaustausch-Communities (ISACs) für kollaboratives Bedrohungsbewusstsein teil
- Korrelieren Sie Threat Intelligence mit dem Asset-Inventar des Legacy-Systems, um anwendbare Bedrohungen zu identifizieren
- Integrieren Sie Threat Intelligence in Sicherheitsmonitoring-Werkzeuge, um Erkennungsfähigkeiten zu verbessern
- Informieren Sie Entwicklungs- und Betriebsteams über Bedrohungen, die spezifisch für ihre Legacy-Technologieplattformen relevant sind
- Nutzen Sie Threat Intelligence, um Patching- und Härtungsaktivitäten basierend auf aktiven Ausnutzungstrends zu priorisieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht proaktive Verteidigung, indem Teams vor Bedrohungen gewarnt werden, bevor sie sich als Vorfälle manifestieren
- Kontextualisiert Sicherheitsinvestitionen, indem gezeigt wird, welche Bedrohungen am relevantesten und aktivsten sind
- Verbessert die Erkennungsgenauigkeit, indem Indikatoren für Kompromittierung für Monitoring-Systeme bereitgestellt werden
- Unterstützt risikobasierte Entscheidungsfindung mit realen Bedrohungsdaten

**Kosten und Risiken:**
- Die Verarbeitung von Threat Intelligence erfordert dedizierte Zeit und analytische Fähigkeiten
- Legacy-Technologie-Stacks könnten im Vergleich zu modernen Plattformen begrenzte Threat-Intelligence-Abdeckung haben
- Informationsüberflutung kann ohne ordentliche Filterung und Priorisierung auftreten
- Threat Intelligence ist vergänglich und erfordert kontinuierliche Updates, um wertvoll zu bleiben

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Versorgungsunternehmen, das Legacy-SCADA-Systeme betrieb, abonnierte einen Threat-Intelligence-Feed für industrielle Steuerungssysteme. Der Feed alarmierte sie über eine aktive Kampagne, die eine spezifische Protokollimplementierung anvisierte, die von ihren Legacy-Controllern genutzt wurde. Weil das Team diese Information erhielt, während die Kampagne noch in ihren frühen Phasen war, konnten sie Netzwerkebenen-Abhilfemaßnahmen implementieren und ein geplantes Firmware-Update beschleunigen, wodurch die Schwachstelle geschlossen wurde, bevor irgendwelche Ausnutzungsversuche ihre Systeme erreichten. Ohne die Threat Intelligence hätten sie von der Kampagne erst nach einem Vorfall oder Monate später durch routinemäßiges Patching erfahren.
