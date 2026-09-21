---
title: Sichere Konfiguration
description: Auslieferung und Betrieb von Systemen mit sicheren
  Standardeinstellungen.
category:
- Security
- Operations
problems:
- configuration-chaos
- configuration-drift
- deployment-environment-inconsistencies
- inadequate-configuration-management
- secret-management-problems
- error-message-information-disclosure
- legacy-configuration-management-chaos
layout: solution
lang: de
en_slug: secure-configuration
related_solutions:
- slug: secure-by-default
  similarity: 0.85
- slug: configuration-checks
  similarity: 0.8
- slug: secure-protocols
  similarity: 0.75
- slug: secure-software-development
  similarity: 0.75
- slug: platform-independent-configuration-management
  similarity: 0.75
- slug: security-certification
  similarity: 0.75
---

## Description

Sichere Konfiguration ist die Praxis, eine sicherheitsgehärtete Konfigurationsbasislinie — die aktivierte Dienste, offene Ports, Standardkonten und Secret-Behandlung abdeckt — für jede Umgebung, in der ein System läuft, zu definieren, zu automatisieren und kontinuierlich zu verifizieren, sodass Produktion, Staging und Entwicklung auf demselben bekannt guten Zustand konvergieren, statt durch unverfolgte manuelle Änderungen auseinanderzudriften. Dies zu erreichen erfordert typischerweise Infrastructure-as-Code-Tooling, um Konfiguration konsistent bereitzustellen, dedizierte Secret-Management-Systeme, um Anmeldedaten vollständig aus Konfigurationsdateien herauszuhalten, und automatisiertes Scanning, das jede Abweichung von der dokumentierten Basislinie bald nach ihrem Auftreten erkennt. Legacy-Systeme sind besonders anfällig für Konfigurationsdrift, weil ihre Umgebungen oft über viele Jahre manuell, von verschiedenen Administratoren angefasst wurden, ohne eine einzige Aufzeichnung dessen, wie die korrekte Konfiguration tatsächlich aussehen soll — ein Zustand, den ein Sicherheitsaudit typischerweise auf die harte Tour aufdeckt, indem es findet, dass Produktionsknoten sich voneinander auf Weisen unterscheiden, die niemand bemerkt oder genehmigt hatte. Ein solches System unter sicheres Konfigurationsmanagement zu bringen bedeutet zunächst, zu dokumentieren, was die Basislinie sein sollte, dann ihre Durchsetzung zu automatisieren, was für Legacy-Komponenten, die nicht für automatisierte Konfiguration entworfen wurden, selbst bedeutsame Tooling-Investition erfordern kann. Einmal etabliert, schließt die Praxis eine der häufigsten Grundursachen von Legacy-Sicherheitsvorfällen — ein versehentlich aktiviertes Debug-Feature oder ein offener Port, der die Erinnerung von irgendjemandem daran, warum er existiert, vordatiert — indem Konfigurationszustand sichtbar, vergleichbar und durchgesetzt statt angenommen wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Erstellen Sie eine Konfigurationsbasislinie, die alle sicherheitsrelevanten Einstellungen für jede Umgebung dokumentiert
- Automatisieren Sie Konfigurationsbereitstellung mittels Infrastructure-as-Code-Werkzeugen, um manuelle Drift zu verhindern
- Entfernen oder deaktivieren Sie alle unnötigen Dienste, Ports und Standardkonten aus Produktionssystemen
- Implementieren Sie Konfigurations-Scanning-Werkzeuge, die Abweichungen von der sicheren Basislinie erkennen
- Trennen Sie Secrets von Konfigurationsdateien und speichern Sie sie in dedizierten Secret-Management-Systemen
- Versionskontrollieren Sie Konfigurationsvorlagen und verlangen Sie Review für jegliche Änderungen
- Führen Sie regelmäßige Konfigurationsaudits durch, die laufende Systeme gegen die dokumentierte Basislinie vergleichen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt häufige Angriffsvektoren, verursacht durch Fehlkonfiguration
- Stellt Konsistenz über Entwicklungs-, Staging- und Produktionsumgebungen sicher
- Bietet Auditierbarkeit und Rückverfolgbarkeit für Konfigurationsänderungen
- Reduziert die Zeit, neue Umgebungen sicher bereitzustellen

**Kosten und Risiken:**
- Legacy-Systeme könnten undokumentierte Konfigurationsabhängigkeiten haben, die bei der Härtung brechen
- Die Automatisierung von Konfiguration für nicht dafür entworfene Systeme kann erhebliche Tooling-Investition erfordern
- Strenges Konfigurationsmanagement kann die Fehlersuche verlangsamen, wenn Entwickler temporär gelockerte Einstellungen brauchen
- Manche Legacy-Komponenten unterstützen möglicherweise keine externalisierte oder automatisierte Konfiguration

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein E-Commerce-Unternehmen, das eine Legacy-.NET-Anwendung betrieb, entdeckte während eines Sicherheitsaudits, dass seine Produktionsserver andere Konfigurationen als Staging hatten, einschließlich aktiviertem Remote-Debugging und ausführlichen Fehlerseiten auf zwei von fünf Produktionsknoten. Das Team erstellte Ansible-Playbooks, die die sichere Konfigurationsbasislinie definierten, und wendete sie über alle Umgebungen an. Automatisierte wöchentliche Scans erkannten jegliche Drift innerhalb von 24 Stunden. Konfigurationsbezogene Sicherheitsbefunde in nachfolgenden Audits fielen von 11 auf einen.
