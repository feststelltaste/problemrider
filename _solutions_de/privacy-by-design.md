---
title: Privacy by Design
description: Einbettung des Datenschutzes in die Systemarchitektur von
  Anfang an.
category:
- Security
- Architecture
problems:
- data-protection-risk
- regulatory-compliance-drift
- authentication-bypass-vulnerabilities
- insecure-data-transmission
- poor-documentation
- fear-of-change
- insufficient-audit-logging
layout: solution
lang: de
en_slug: privacy-by-design
related_solutions:
- slug: security-by-design
  similarity: 0.75
- slug: security-architecture-analysis
  similarity: 0.75
- slug: secure-protocols
  similarity: 0.7
- slug: threat-modeling
  similarity: 0.7
- slug: encryption
  similarity: 0.7
- slug: data-strategy
  similarity: 0.7
---

## Description

Privacy by Design ist die Praxis, Datenschutz von Anfang an als architektonische Beschränkung zu behandeln — Minimierung der erfassten personenbezogenen Daten, deren Verschlüsselung oder Pseudonymisierung entsprechend der Sensibilität, Beschränkung des Zugangs nach Rolle und Einbau von Aufbewahrungs- und Löschrichtlinien —, statt Datenschutz als eine Compliance-Checkliste zu behandeln, die nach dem Bau des Systems angewendet wird. Auf ein bestehendes Legacy-System angewendet, wird dies notwendigerweise zu einem nachträglichen Einbau: eine vollständige Inventarisierung, welche personenbezogenen Daten wo existieren, gefolgt vom Hinzufügen der Verschlüsselung, Zugangskontrolle, Einwilligungsverfolgung und automatisierten Aufbewahrungsmechanismen, die von Anfang an eingeplant worden wären, hätte die regulatorische Landschaft zum Zeitpunkt des Systembaus existiert. Dies ist in Legacy-Kontexten besonders kostspielig, weil personenbezogene Daten in älteren Systemen oft undokumentiert über viele Tabellen und Integrationen verstreut sind, angesammelt über Jahre ad hoc vorgenommener Schemaänderungen ohne zentralisierte Datenkarte, sodass allein der anfängliche Prüfschritt ein erhebliches Unterfangen sein kann, bevor irgendeine technische Behebung beginnt. Der Aufwand wird dennoch üblicherweise durch externen Druck erzwungen — eine Regulierung wie die DSGVO, die in Kraft tritt, oder eine Datenschutzverletzung —, weil das Fehlen von Datenschutzkontrollen in einem jahrzehntealten System sowohl akute regulatorische Exposition als auch eine breite, weitgehend undokumentierte Angriffsfläche darstellt. Der Zielkonflikt ist, dass Datenminimierung und Anonymisierung, rückwirkend angewendet, mit bestehenden Geschäftsprozessen und Support-Arbeitsabläufen kollidieren können, die unter der Annahme vollständigen, uneingeschränkten Zugangs zu historischen personenbezogenen Daten gebaut wurden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Führen Sie eine Dateninventur durch, um alle vom Legacy-System erfassten, gespeicherten und verarbeiteten personenbezogenen Daten zu identifizieren
- Klassifizieren Sie Daten nach Sensibilitätsstufe und wenden Sie angemessene Verschlüsselungs-, Anonymisierungs- oder Pseudonymisierungstechniken an
- Implementieren Sie Datenminimierung, indem Sie unnötige Datenerfassungspunkte aus Legacy-Formularen und APIs entfernen
- Fügen Sie Einwilligungsverwaltungsmechanismen und Audit-Trails für Datenverarbeitungsaktivitäten hinzu
- Führen Sie Datenaufbewahrungsrichtlinien mit automatisierter Löschung abgelaufener personenbezogener Daten ein
- Rüsten Sie Zugangskontrollen nach, um sicherzustellen, dass personenbezogene Daten nur für autorisierte Rollen zugänglich sind
- Dokumentieren Sie Datenflüsse zwischen Legacy-Komponenten und Drittanbieter-Integrationen, um Datenschutzrisiken zu identifizieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert regulatorisches Risiko und mögliche Bußgelder durch Datenschutzverletzungen
- Baut Nutzervertrauen auf, indem Engagement für Datenschutz demonstriert wird
- Vereinfacht künftige Compliance-Anstrengungen, indem Datenschutz als grundlegendes Anliegen etabliert wird
- Reduziert die Angriffsfläche, indem die Menge gespeicherter sensibler Daten begrenzt wird

**Kosten und Risiken:**
- Datenschutz nachträglich in Legacy-Systeme einzubauen ist erheblich teurer als ihn von Anfang an einzubauen
- Datenminimierung kann Änderungen an Geschäftsprozessen erfordern, die auf historischen Daten beruhen
- Anonymisierung und Pseudonymisierung können Debugging- und Support-Arbeitsabläufe komplizieren
- Einwilligungsverwaltung fügt nutzerseitigen Oberflächen und Backend-Logik Komplexität hinzu

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Anfang der 2000er-Jahre gebaute Gesundheitsplattform speicherte Patientendaten im Klartext über mehrere Datenbanktabellen hinweg ohne Zugangskontrollen jenseits anwendungsseitiger Authentifizierung. Als die DSGVO in Kraft trat, führte das Team eine vollständige Datenprüfung durch und entdeckte personenbezogene Daten in 47 Tabellen über drei Datenbanken. Sie führten spaltenebenen Verschlüsselung für sensible Felder ein, fügten rollenbasierte Zugangskontrollen hinzu und implementierten automatisierte Datenaufbewahrung mit konfigurierbaren Löschplänen. Das Projekt dauerte acht Monate, reduzierte aber ihre Compliance-Risikoexposition von kritisch auf niedrig.
