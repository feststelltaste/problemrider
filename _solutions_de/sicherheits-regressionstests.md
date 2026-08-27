---
title: Sicherheits-Regressionstests
description: Erneutes Testen zuvor behobener Sicherheitslücken, um ihr
  Wiederauftreten zu verhindern.
category:
- Security
- Testing
problems:
- regression-bugs
- insufficient-testing
- legacy-code-without-tests
- high-bug-introduction-rate
- fear-of-breaking-changes
- partial-bug-fixes
- test-debt
- poor-test-coverage
layout: solution
lang: de
en_slug: regression-tests
related_solutions:
- slug: security-tests
  similarity: 0.85
- slug: security-tests-by-external-parties
  similarity: 0.8
- slug: vulnerability-scans
  similarity: 0.8
- slug: security-audits
  similarity: 0.8
- slug: secure-software
  similarity: 0.8
- slug: secure-software-development
  similarity: 0.8
---

## Description

Sicherheits-Regressionstests sind automatisierte Testfälle, die spezifisch eine zuvor entdeckte und behobene Schwachstelle reproduzieren, sodass genau der Angriffsvektor, der einst erfolgreich war, bei jedem Build erneut gegen das System versucht wird, um zu bestätigen, dass der Fix noch hält. Anders als allgemeine funktionale Regressionstests werden sie direkt aus einem Schwachstellenbericht oder einem Penetrationstest-Befund geschrieben und kodieren die spezifische Payload, Eingabe oder Anfragefolge, die zuvor eine Sicherheitskontrolle umging. Dies zählt in Legacy-Systemen, weil Sicherheitsfixes dort oft als enge, lokalisierte Patches auf Code angewendet werden, der sonst schlecht verstanden, schlecht getestet und häufigen, unkoordinierten Änderungen durch verschiedene Teams unterworfen ist — Bedingungen, unter denen eine behobene Schwachstelle stark anfällig dafür ist, wieder aufzutauchen, sei es durch eine Rückabwicklung, einen parallelen Codepfad, der unabhängig denselben Fehler erhielt, oder ein Refactoring, das versehentlich den ursprünglichen Fix rückgängig macht. Durch das kontinuierliche Ausführen dieser Tests in der CI/CD-Pipeline statt sich auf periodische manuelle Penetrationstests zu verlassen, erhalten Teams sofortiges Feedback in dem Moment, in dem eine Änderung eine bekannte Schwäche wieder einführt, und schließen die Lücke zwischen der Einführung einer Regression und ihrer Erfassung. Über die Zeit wird die wachsende Suite von Sicherheits-Regressionstests zu einem ausführbaren Datensatz der Sicherheitsgeschichte des Systems, der institutionelles Wissen erfasst, das sonst davon abhängen würde, dass sich Einzelpersonen an Vorfälle aus vergangenen Jahren erinnern.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Erstellen Sie eine dedizierte Sicherheits-Regressionstest-Suite, die für jede zuvor identifizierte und behobene Schwachstelle einen Testfall enthält
- Integrieren Sie Sicherheits-Regressionstests in die CI/CD-Pipeline, sodass sie bei jedem Build laufen
- Schreiben Sie Tests, die spezifisch den ursprünglichen Angriffsvektor reproduzieren, um zu bestätigen, dass der Fix weiterhin wirksam ist
- Pflegen Sie ein Schwachstellenregister, das jeden Befund seinem entsprechenden Regressionstest zuordnet
- Erweitern Sie Regressionstests, wenn neue Angriffsvarianten oder Umgehungstechniken entdeckt werden
- Überprüfen und aktualisieren Sie Sicherheits-Regressionstests, wenn der betroffene Code refaktoriert wird

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verhindert die Wiedereinführung zuvor behobener Sicherheitslücken während Refactoring oder Feature-Entwicklung
- Baut institutionelles Gedächtnis vergangener Sicherheitsprobleme in ausführbarer Form auf
- Bietet Vertrauen, dass Änderungen am Legacy-System die Sicherheitslage nicht still verschlechtern
- Ergänzt manuelle Sicherheitstests durch Automatisierung der Verifikation bekannter Probleme

**Kosten und Risiken:**
- Bedeutsame Sicherheits-Regressionstests zu schreiben erfordert Verständnis sowohl der Schwachstelle als auch des Fixes
- Die Testsuite-Pflege wächst über die Zeit und kann CI-Pipelines verlangsamen, wenn nicht gemanagt
- Tests können falsches Vertrauen geben, wenn sie die ursprünglichen Angriffsbedingungen nicht genau reproduzieren
- Legacy-Systeme mit schlechter Testbarkeit könnten erhebliches Refactoring erfordern, bevor Tests hinzugefügt werden können

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Bankanwendung hatte über zwei Jahre drei separate Wiederauftritte einer Cross-Site-Scripting-Schwachstelle erlebt, jedes Mal in einem leicht unterschiedlichen Eingabefeld. Nach dem dritten Auftreten erstellte das Team eine Sicherheits-Regressionstest-Suite, die browserbasierte Injektionsversuche gegen alle Nutzereingabefelder automatisierte. Jeder neue Schwachstellenbefund wurde sofort als Regressionstest hinzugefügt. Im folgenden Jahr erfasste die Regressionssuite zwei zusätzliche Fälle, in denen Entwickler während der Feature-Entwicklung versehentlich ähnliche Schwachstellen einführten, und verhinderte, dass sie Produktion erreichten.
