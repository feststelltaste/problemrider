---
title: Secure by Default
description: Ausrichtung von Standardeinstellungen und Auslieferungszustand
  auf maximale Sicherheit.
category:
- Security
- Operations
problems:
- configuration-chaos
- configuration-drift
- secret-management-problems
- password-security-weaknesses
- authentication-bypass-vulnerabilities
- error-message-information-disclosure
- inadequate-configuration-management
layout: solution
lang: de
en_slug: secure-by-default
related_solutions:
- slug: secure-configuration
  similarity: 0.85
- slug: secure-protocols
  similarity: 0.8
- slug: secure-software-development
  similarity: 0.75
- slug: secure-programming-interfaces
  similarity: 0.75
- slug: secure-coding-guidelines
  similarity: 0.75
- slug: secure-software
  similarity: 0.75
---

## Description

Secure by Default bedeutet, ein System so auszuliefern und zu konfigurieren, dass seine Out-of-the-Box-Einstellungen bereits die restriktivsten sind, die mit der Funktionsfähigkeit des Systems vereinbar sind, statt einen Administrator zu erfordern, es nach der Installation aktiv zu härten — unnötige Dienste und Debug-Endpunkte deaktivieren, starke Standardanmeldedaten oder gar keine nutzen, und sicherstellen, dass Fehlermeldungen nie interne Details wie Stack Traces oder Verbindungsstrings preisgeben. Das zugrunde liegende Prinzip ist, dass Sicherheit nicht davon abhängen sollte, dass sich jeder Betreiber erinnert und einen Härtungsschritt korrekt durchführt, da in der Praxis ein Teil der Bereitstellungen diesen Schritt immer auslassen wird. Legacy-Systeme sind hier besonders exponiert, weil viele von ihnen zu einer Zeit gebaut oder konfiguriert wurden, als unsichere Standardeinstellungen — ausführliche Debug-Ausgabe, Standard-Admin-Passwörter, offene Diagnoseports — die Branchennorm statt die Ausnahme waren, und diese Standardeinstellungen haben oft jahrelang unangetastet fortbestanden, einfach weil niemand die ursprüngliche Installation überprüft hat. Sichere-Standard-Einstellungen nachträglich in ein solches System einzubauen bedeutet, zu prüfen, was die aktuellen Standardeinstellungen tatsächlich sind, was häufig vergessene Konfigurationen offenlegt, die niemand wissentlich genehmigt hätte, und dann ein gehärtetes Konfigurationsprofil zu bauen, das die neue Basislinie für jede künftige Umgebung wird. Da Legacy-Systeme undokumentierte Abhängigkeiten von genau den unsicheren Verhaltensweisen haben können, die entfernt werden, muss die Änderung von Standardeinstellungen sorgfältig ausgerollt werden, aber die resultierende Reduktion der Angriffsfläche gilt automatisch für jede künftige Bereitstellung, ohne auf anhaltende administrative Wachsamkeit angewiesen zu sein.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Prüfen Sie alle Standardkonfigurationen im Legacy-System auf unsichere Einstellungen wie Standardpasswörter, offene Ports und ausführliche Fehlermeldungen
- Ändern Sie Standardeinstellungen auf die restriktivsten Optionen, die dem System noch erlauben zu funktionieren
- Deaktivieren Sie unnötige Features, Dienste und Debug-Endpunkte in Produktionsbereitstellungen
- Stellen Sie sicher, dass Fehlermeldungen keine internen Systemdetails wie Stack Traces, Versionsnummern oder Dateipfade preisgeben
- Liefern Sie Konfigurationsvorlagen mit sicherheitsgehärteten Standardeinstellungen aus und verlangen Sie explizites Opt-in für gelockerte Einstellungen
- Dokumentieren Sie die Sicherheitsbegründung für jede Standardeinstellung, sodass künftige Betreuer verstehen, warum sie gewählt wurde

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert die Angriffsfläche, ohne laufende Nutzeraktion zu erfordern
- Verhindert häufige Fehlkonfigurationen, die zu Sicherheitsvorfällen führen
- Senkt die Hürde für sichere Bereitstellung, indem Sicherheit zum Weg des geringsten Widerstands wird
- Erfasst Versäumnisse, bei denen Administratoren vergessen, nicht standardmäßige Installationen zu härten

**Kosten und Risiken:**
- Zu restriktive Standardeinstellungen können bestehende Integrationen oder Arbeitsabläufe brechen, die von gelockerten Einstellungen abhängen
- Nutzer könnten sichere Standardeinstellungen umgehen statt zu verstehen, warum sie existieren
- Legacy-Systeme haben oft undokumentierte Abhängigkeiten von unsicheren Standardverhaltensweisen
- Die Änderung von Standardeinstellungen in Produktionssystemen erfordert sorgfältiges Rollout und Testing

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein SaaS-Unternehmen entdeckte, dass sein Legacy-Anwendungsserver standardmäßig mit aktiviertem Debug-Modus ausgeliefert wurde, was detaillierte Stack Traces und Datenbankverbindungsstrings in Fehlerantworten offenlegte. Ein Sicherheitsaudit fand außerdem, dass das Standard-Admin-Konto ein wohlbekanntes Passwort nutzte. Das Team erstellte ein gehärtetes Konfigurationsprofil, das den Debug-Modus deaktivierte, starke initiale Passwörter erzwang und unnötige Netzwerkports schloss. Nach der Bereitstellung der neuen Standardeinstellungen über alle Umgebungen hinweg fiel die Anzahl der Informationsoffenlegungsbefunde in nachfolgenden Penetrationstests von 14 auf null.
