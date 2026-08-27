---
title: Runbooks
description: Bereitstellung detaillierter Anweisungen zur Bearbeitung von
  Aufgaben und Vorfällen.
category:
- Operations
- Communication
problems:
- slow-incident-resolution
- knowledge-silos
- knowledge-dependency
- implicit-knowledge
- poor-documentation
- constant-firefighting
- difficult-developer-onboarding
- inconsistent-execution
- change-management-chaos
- no-formal-change-control-process
layout: solution
lang: de
en_slug: runbooks
related_solutions:
- slug: incident-management
  similarity: 0.85
- slug: checklists
  similarity: 0.8
- slug: knowledge-sharing-practices
  similarity: 0.75
- slug: root-cause-analysis
  similarity: 0.75
- slug: security-incident-handling
  similarity: 0.75
- slug: living-documentation
  similarity: 0.75
---

## Description

Ein Runbook ist ein schriftliches, schrittweises Verfahren zur Behandlung einer spezifischen Betriebsaufgabe oder eines Vorfalls — einschließlich Diagnoseschritten, Behebungsaktionen, Eskalationspfaden und Rollback-Anweisungen —, geschrieben auf einem Detailniveau, das jemandem, der mit dem System nicht vertraut ist, erlaubt, es unter Druck erfolgreich zu befolgen. Runbooks sind am nützlichsten, wenn sie an einem durchsuchbaren, versionskontrollierten Ort gespeichert, direkt mit den für jedes Verfahren relevanten Monitoring-Dashboards und Log-Abfragen verknüpft und unmittelbar nach jedem Vorfall aktualisiert werden, der eine Lücke in der bestehenden Dokumentation aufdeckte. In Legacy-Systemen existiert betriebliches Wissen darüber, wie genau wiederkehrende Fehlermodi diagnostiziert und behoben werden, sehr oft nur in den Köpfen von ein oder zwei langjährigen Ingenieuren, die das System gebaut oder jahrelang gewartet haben, was einen schweren Single Point of Failure in der Organisation selbst schafft: Wenn diese Person während eines Vorfalls nicht verfügbar ist, kann sich die Behebungszeit von Minuten auf Stunden strecken, selbst für Probleme, die im Prinzip gut verstanden sind. Das Schreiben von Runbooks verwandelt dieses stillschweigende, personengebundene Wissen in einen expliziten, teilbaren Vermögenswert, was direkt sowohl die Vorfallbehebungszeit als auch die Abhängigkeit der Organisation von spezifischen Personen reduziert, um ein Legacy-System am Laufen zu halten. Die Investition erfordert jedoch laufende Pflege, da ein Runbook, das aus dem Gleichschritt mit dem tatsächlichen Systemverhalten geraten ist, aktiv in die Irre führen kann, wer auch immer es während eines nachfolgenden Vorfalls befolgt, was veraltete Runbooks in mancher Hinsicht schlimmer macht als gar keine Runbooks zu haben.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Dokumentieren Sie schrittweise Verfahren für alle bekannten Legacy-System-Fehlermodi und häufigen Betriebsaufgaben
- Beziehen Sie Diagnoseschritte, Behebungsverfahren, Eskalationspfade und Rollback-Anweisungen ein
- Speichern Sie Runbooks in einem durchsuchbaren, versionskontrollierten System, zugänglich für alle Bereitschaftsingenieure
- Schreiben Sie Runbooks auf einem Niveau, das jemandem, der mit dem Legacy-System nicht vertraut ist, erlaubt, sie zu befolgen
- Aktualisieren Sie Runbooks nach jedem Vorfall, bei dem die bestehende Dokumentation unzureichend war
- Beziehen Sie Links zu Monitoring-Dashboards, Log-Abfragen und Konfigurationsorten ein, die für jedes Verfahren relevant sind
- Überprüfen und testen Sie Runbooks periodisch, um sicherzustellen, dass sie akkurat bleiben, während sich das System weiterentwickelt
- Weisen Sie jedem Runbook Eigentümerschaft zu, um Verantwortlichkeit für ihre Aktualität sicherzustellen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert die durchschnittliche Behebungszeit durch sofortige Anleitung während Vorfällen
- Erfasst institutionelles Wissen über Legacy-Systeme, das sonst nur in den Köpfen von Personen existieren würde
- Ermöglicht weniger erfahrenen Teammitgliedern, Vorfälle selbstbewusst zu handhaben
- Reduziert die Abhängigkeit von spezifischen Personen für betriebliches Legacy-System-Wissen

**Kosten und Risiken:**
- Runbooks erfordern laufenden Pflegeaufwand, um aktuell zu bleiben
- Übermäßig starre Runbooks können kritisches Denken während neuartiger Vorfälle entmutigen
- Veraltete Runbooks können falsche Anleitung liefern, die Vorfälle verschlimmert
- Das Schreiben umfassender Runbooks erfordert erheblichen anfänglichen Aufwand

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Abrechnungssystem eines Finanzunternehmens hatte Betriebsverfahren, die nur zwei leitenden Ingenieuren bekannt waren, die es über ein Jahrzehnt gewartet hatten. Als ein Ingenieur ging und der andere während eines kritischen Vorfalls im Urlaub war, verbrachte das Bereitschaftsteam vier Stunden mit der Diagnose eines Problems, das normalerweise 15 Minuten zur Behebung brauchte. Das Team investierte anschließend zwei Wochen in die Erstellung von Runbooks für die 20 häufigsten Vorfalltypen, einschließlich Datenbankverbindungs-Reset-Verfahren, Batch-Job-Neustartsequenzen und Datenabgleichsschritte. Der nächste ähnliche Vorfall wurde in 12 Minuten von einem Junior-Ingenieur gelöst, der dem Runbook folgte.
