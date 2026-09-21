---
title: Test-Driven Development (TDD)
description: Schreiben von Tests vor der eigentlichen Implementierung.
category:
- Code
- Testing
problems:
- legacy-code-without-tests
- poor-test-coverage
- regression-bugs
- difficult-to-test-code
- fear-of-change
- high-bug-introduction-rate
- refactoring-avoidance
layout: solution
lang: de
en_slug: test-driven-development-tdd
related_solutions:
- slug: automated-tests
  similarity: 0.75
- slug: mutation-testing
  similarity: 0.75
- slug: integration-tests
  similarity: 0.75
- slug: dependency-injection
  similarity: 0.7
- slug: living-documentation
  similarity: 0.7
- slug: trunk-based-development
  similarity: 0.7
---

## Description

Test-Driven Development schreibt einen fehlschlagenden Test vor jeglichem Implementierungscode, bringt ihn mit der minimal nötigen Änderung zum Bestehen und refaktoriert dann — ein Rot-Grün-Refaktorieren-Zyklus, der Testbarkeit selbst als Designbeschränkung nutzt statt als nachträglichen Gedanken, angewendet, sobald der Code bereits existiert. TDD auf eine ganze Legacy-Codebasis auf einmal nachzurüsten ist selten realistisch, sodass die Praxis üblicherweise zuerst auf neuen Code und Fehlerbehebungen angewendet wird, gepaart mit Charakterisierungstests, die bestehendes Verhalten erfassen, bevor überhaupt Legacy-Code unter Modifikation angefasst wird. Da Code, der schwer zu testen ist, sehr oft ein Zeichen verworrener Abhängigkeiten und schlechter Belangstrennung ist, neigt die Disziplin, den Test zuerst zu schreiben, dazu, genau die Designprobleme zutage zu bringen und unter Druck zu setzen, die Legacy-Code überhaupt erst brüchig machen, auf Kosten eines anfänglichen Produktivitätseinbruchs, während ein Team echte Kompetenz mit dem Zyklus aufbaut.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Folgen Sie dem Rot-Grün-Refaktorieren-Zyklus: Schreiben Sie einen fehlschlagenden Test, bringen Sie ihn mit minimalem Code zum Bestehen, refaktorieren Sie dann
- Schreiben Sie beim Modifizieren von Legacy-Code zuerst Charakterisierungstests, um bestehendes Verhalten zu erfassen, bevor Änderungen vorgenommen werden
- Beginnen Sie mit der Anwendung von TDD auf neuen Code und Fehlerbehebungen, statt zu versuchen, die gesamte Legacy-Codebasis nachzurüsten
- Nutzen Sie TDD als Designwerkzeug: Wenn Code schwer zu testen ist, braucht das Design wahrscheinlich Verbesserung
- Halten Sie Testzyklen kurz (unter ein paar Minuten pro Rot-Grün-Refaktorieren-Zyklus), um den Fluss aufrechtzuerhalten
- Paaren Sie TDD mit Refactoring: Nachdem Tests bestehen, verbessern Sie das Design, während das Sicherheitsnetz vorhanden ist
- Bauen Sie Teamfähigkeiten durch Coding-Dojos und Pair-Programming-Sitzungen auf, die sich auf TDD fokussieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Produziert Code mit von Anfang an eingebauter Testabdeckung
- Treibt einfachere, modularere Designs voran, weil Testbarkeit eine Designbeschränkung ist
- Bietet unmittelbares Feedback dazu, ob Codeänderungen bestehendes Verhalten brechen
- Reduziert Debugging-Zeit durch Erfassung von Defekten im Moment ihrer Einführung

**Kosten und Risiken:**
- Erfordert erhebliche Übung, um kompetent zu werden; die anfängliche Produktivität könnte sinken
- Nicht jeder Legacy-Code ist für TDD zugänglich, ohne zuerst Abhängigkeiten zu extrahieren
- Kann zu Übertestung von Implementierungsdetails führen, wenn Eigenschaften nicht gut gewählt sind
- Teams unter starkem Fristendruck könnten die Praxis aufgeben, bevor sie die Vorteile erkennen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Team, das ein Legacy-Lohnabrechnungssystem pflegte, wurde beauftragt, Unterstützung für eine neue Steuervorschrift hinzuzufügen. Statt den bestehenden ungetesteten Steuerberechnungscode direkt zu modifizieren, schrieben sie Charakterisierungstests, um das aktuelle Verhalten des Moduls zu erfassen. Dann, mit TDD, schrieben sie fehlschlagende Tests für die neue Vorschrift, implementierten die Logik, um sie zum Bestehen zu bringen, und refaktorierten das Ergebnis. Der Prozess dauerte etwas länger als der übliche Ansatz des Teams, aber das Modul wurde mit umfassender Testabdeckung ausgeliefert. Als zwei Monate später eine Folgeänderung der Vorschrift eintraf, nahm das Team die Modifikation zuversichtlich in der halben Zeit vor, geleitet von der bestehenden Testsuite.
