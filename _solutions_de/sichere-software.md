---
title: Sichere Software
description: Verhinderung von Zuverlässigkeitsvorfällen, die durch
  Sicherheitslücken verursacht werden.
category:
- Security
problems:
- authentication-bypass-vulnerabilities
- buffer-overflow-vulnerabilities
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- secret-management-problems
- password-security-weaknesses
- data-protection-risk
layout: solution
lang: de
en_slug: secure-software
related_solutions:
- slug: chaos-engineering
  similarity: 0.8
- slug: incident-management
  similarity: 0.8
- slug: security-tests
  similarity: 0.8
- slug: production-environment-maintenance
  similarity: 0.8
- slug: static-code-analysis
  similarity: 0.8
- slug: secure-protocols
  similarity: 0.8
---

## Description

Sichere Software ist in diesem Kontext die Menge von Praktiken, die bekannte, ausnutzbare Schwachstellenklassen — Injection-Fehler, Authentifizierungsumgehungen, Buffer Overflows, fest codierte Secrets — aus einer bestehenden Codebasis entfernt, bevor sie ausgelöst werden und einen Zuverlässigkeitsvorfall verursachen können, statt Sicherheitsdefekte rein als Compliance-Anliegen getrennt von Systemstabilität zu behandeln. Der zugrunde liegende Mechanismus ist, dass Sicherheitsschwachstellen und Zuverlässigkeitsvorfälle in der Praxis keine getrennten Kategorien sind: eine ausgenutzte SQL-Injection oder ein ausgelöster Buffer Overflow produziert denselben Ausfall, Datenverlust oder Verfügbarkeitseinfluss wie jeder andere schwerwiegende Defekt, er kommt nur über einen feindseligen statt versehentlichen Pfad. Legacy-Code ist hier überproportional exponiert, weil er oft geschrieben wurde, bevor sichere Programmierpraktiken breit gelehrt oder in Werkzeuge integriert wurden, seine Abhängigkeiten haben bekannte CVEs schneller angesammelt, als sie irgendjemand gepatcht hat, und seine Authentifizierungs- und Verschlüsselungsmechanismen könnten Standards vordatieren, die heute als Grundlinie gelten. Diese Lösung anzuwenden bedeutet, Sicherheitsaudits und statische Analyse spezifisch gegen die Legacy-Codebasis durchzuführen, um diese Muster offenzulegen, dann sie durch Input-Validierung, parametrisierte Abfragen, Abhängigkeits-Updates und ordentliches Secret Management zu schließen, wobei jeder Fix als Zuverlässigkeitsverbesserung behandelt wird, nicht nur als Sicherheitsverbesserung. Da Legacy-Systemen oft die Testabdeckung fehlt, die solche Änderungen risikoarm machen würde, muss die Behebung sorgfältig sequenziert werden, aber der Gewinn ist, dass sie dieselben Grundursachen adressiert, die sonst als unvorhersehbare Ausfälle wieder auftauchen würden, unabhängig davon, ob sie als Sicherheits- oder Stabilitätsprobleme gerahmt werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Führen Sie Sicherheitsaudits von Legacy-Code durch, um bekannte Schwachstellenmuster zu identifizieren (Injection, Authentifizierungsumgehung usw.)
- Wenden Sie Sicherheits-Patches zeitnah für alle von Legacy-Systemen genutzten Frameworks, Bibliotheken und Laufzeitumgebungen an
- Implementieren Sie Input-Validierung und Output-Encoding an Systemgrenzen, um Injection-Angriffe zu verhindern
- Fügen Sie Abhängigkeitsscanning zu CI/CD-Pipelines hinzu, um bekannte Schwachstellen in Legacy-Abhängigkeiten zu erkennen
- Migrieren Sie von veralteten Authentifizierungs- und Verschlüsselungsmechanismen zu aktuellen Standards
- Implementieren Sie ordentliches Secret Management, um fest codierte Anmeldedaten aus Legacy-Codebasen zu entfernen
- Nutzen Sie statische Sicherheitsanalysewerkzeuge (SAST), konfiguriert für den Legacy-Technologie-Stack

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verhindert Zuverlässigkeitsvorfälle, verursacht durch Ausnutzung von Sicherheitslücken
- Schützt Geschäftsreputation und Kundenvertrauen
- Reduziert Compliance-Risiko für Legacy-Systeme, die sensible Daten handhaben
- Sicherheitsverbesserungen verbessern oft die gesamte Codequalität

**Kosten und Risiken:**
- Sicherheitsbehebung in Legacy-Code kann ohne gute Testabdeckung zeitaufwendig und riskant sein
- Die Aktualisierung von Authentifizierungs- oder Verschlüsselungsmechanismen kann bestehende Integrationen brechen
- Sicherheitsscanning-Werkzeuge könnten viele falsch positive Ergebnisse für Legacy-Code-Muster produzieren
- Manche Legacy-Schwachstellen könnten erhebliche architektonische Änderungen zur Behebung erfordern

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Patientenportal einer Gesundheitsorganisation wurde zwei Tage lang offline genommen, nachdem eine SQL-Injection-Schwachstelle ausgenutzt wurde, was einen Zuverlässigkeitsvorfall verursachte, der Tausende von Patienten betraf. Nach dem Vorfall führte das Team ein umfassendes Sicherheitsaudit durch, das 15 Injection-Punkte, fest codierte Datenbankanmeldedaten und eine veraltete Authentifizierungsbibliothek mit bekannten Umgehungen offenbarte. Durch die Implementierung parametrisierter Abfragen, die Migration zu einem aktuellen Authentifizierungs-Framework und das Hinzufügen automatisierten Abhängigkeitsscannens beseitigte das Team die Schwachstellenklassen, die den Ausfall verursacht hatten, und verhinderte künftig ähnliche Zuverlässigkeitsvorfälle.
