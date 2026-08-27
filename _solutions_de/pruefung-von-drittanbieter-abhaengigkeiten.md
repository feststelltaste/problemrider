---
title: Prüfung von Drittanbieter-Abhängigkeiten
description: Regelmäßige Überprüfung von Abhängigkeiten von externer
  Software.
category:
- Security
- Dependencies
problems:
- dependency-version-conflicts
- obsolete-technologies
- vendor-dependency
- shared-dependencies
- technology-lock-in
- high-technical-debt
- breaking-changes
- dependency-on-supplier
layout: solution
lang: de
en_slug: third-party-dependency-check
related_solutions:
- slug: vulnerability-scans
  similarity: 0.8
- slug: dependency-management-strategy
  similarity: 0.8
- slug: regular-maintenance-and-updates
  similarity: 0.8
- slug: secure-software
  similarity: 0.75
- slug: security-audits
  similarity: 0.75
- slug: security-tests
  similarity: 0.75
---

## Description

Eine Prüfung von Drittanbieter-Abhängigkeiten scannt kontinuierlich die externen Abhängigkeiten eines Systems — mit Werkzeugen wie OWASP Dependency-Check, Snyk oder Dependabot — gegen bekannte Schwachstellendatenbanken und Wartungsstatussignale, statt darauf zu vertrauen, dass eine Bibliothek, die heute korrekt funktioniert, auch weiterhin sicher und unterstützt ist. Legacy-Systeme sammeln über Jahre Abhängigkeiten an, die still ungepflegt oder verwundbar werden, ohne dass es jemand bemerkt, genau weil eine Bibliothek ohne aktive Probleme kein äußeres Signal gibt, dass ihr Maintainer verschwunden ist oder dass eine kritische Schwachstelle gegen sie bekannt wurde. Diese Prüfung innerhalb der CI/CD-Pipeline zu automatisieren, gestützt durch eine klare Richtlinie dafür, wie schnell bekannte Schwachstellen adressiert werden müssen, verwandelt eine leicht vernachlässigte manuelle Aufgabe in eine routinemäßige — obwohl Legacy-Systeme mit tief eingebetteten, schwer aufzurüstenden Abhängigkeiten weiterhin einem echten Risiko kaskadierender Versionskonflikte ausgesetzt sind, sobald ein Update tatsächlich versucht wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Implementieren Sie automatisierte Abhängigkeits-Scanning-Werkzeuge (z. B. OWASP Dependency-Check, Snyk, Dependabot) in der CI/CD-Pipeline
- Pflegen Sie ein Inventar aller Drittanbieter-Abhängigkeiten einschließlich Version, Lizenz und Wartungsstatus
- Etablieren Sie eine Richtlinie für maximal akzeptables Schwachstellenalter und -schweregrad vor verpflichtenden Updates
- Planen Sie regelmäßige Abhängigkeits-Review-Sitzungen, um die Gesundheit und Sicherheitslage kritischer Abhängigkeiten zu bewerten
- Erstellen Sie Upgrade-Pfade für Abhängigkeiten, die das Ende ihrer Lebensdauer erreicht haben oder keine Sicherheitspatches mehr erhalten
- Überwachen Sie Abhängigkeitsprojekte auf Anzeichen von Aufgabe, Eigentümerwechsel oder kompromittierten Releases
- Testen Sie Abhängigkeits-Updates in isolierten Umgebungen, bevor sie in Produktion ausgerollt werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Identifiziert bekannte Schwachstellen in Abhängigkeiten, bevor sie ausgenutzt werden
- Bietet Sichtbarkeit in die Sicherheits- und Wartungsgesundheit des Abhängigkeitsportfolios
- Ermöglicht proaktive Planung für Abhängigkeitsmigrationen, bevor sie dringend werden
- Reduziert das Risiko der Nutzung aufgegebener oder kompromittierter Bibliotheken

**Kosten und Risiken:**
- Legacy-Systeme haben oft tief eingebettete Abhängigkeiten, die schwer ohne Breaking Changes zu aktualisieren sind
- Automatisierte Scanner decken möglicherweise nicht alle Abhängigkeitstypen ab, besonders maßgeschneiderte oder vendored Bibliotheken
- Häufige Abhängigkeits-Updates können Regressionen einführen, wenn sie nicht ordentlich getestet werden
- Die Aktualisierung einer Abhängigkeit in einem Legacy-System kann kaskadierende Versionskonflikte auslösen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Content-Management-System eines Medienunternehmens nutzte eine JSON-Parsing-Bibliothek, die drei Jahre zuvor von ihrem Maintainer aufgegeben worden war. Eine automatisierte Abhängigkeitsprüfung markierte diese Bibliothek als mit zwei ungepatchten Schwachstellen hoher Schwere. Da das Team regelmäßige Abhängigkeits-Reviews durchgeführt hatte, hatten sie bereits einen Migrationspfad zu einer unterstützten Alternative identifiziert und den Wechsel innerhalb eines Sprints abgeschlossen. Ohne die automatisierte Prüfung wären die Schwachstellen unentdeckt geblieben, da die Bibliothek korrekt funktionierte und niemand im Team ihren Sicherheitsstatus manuell überwachte.
